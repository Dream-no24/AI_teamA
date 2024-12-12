import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path

# YOLOv5 경로를 autoLabeling.py 위치 기준으로 추가
CURRENT_DIR = Path(__file__).resolve().parent  # autoLabeling.py가 위치한 디렉터리
YOLOV5_PATH = CURRENT_DIR / "yolov5"  # yolov5 경로를 기준으로 설정
sys.path.insert(0, str(YOLOV5_PATH))

# YOLOv5 모듈 임포트
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, box_iou
from utils.augmentations import letterbox  # Letterbox 추가

def load_model(model_path):
    device = select_device("")  # GPU/CPU 자동 선택
    model = DetectMultiBackend(model_path, device=device)  # 모델 로드
    model.warmup()  # 워밍업 (옵션)
    return model

def load_existing_labels(label_path):
    """기존 텍스트 파일에서 라벨을 불러옵니다."""
    if not os.path.exists(label_path):
        return []  # 파일이 없으면 빈 리스트 반환

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls, x_center, y_center, width, height = map(float, parts)
            labels.append([cls, x_center, y_center, width, height])
    return labels

def calculate_iou(box1, box2):
    """IoU 계산 함수"""
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def merge_boxes(detections, iou_threshold=0.5):
    """겹치는 박스를 통합하는 함수"""
    if len(detections) == 0:
        return []
    
    # numpy 배열로 변환
    detections = np.array(detections)
    
    try:
        x_centers = detections[:, 1]
        y_centers = detections[:, 2]
        widths = detections[:, 3]
        heights = detections[:, 4]
    except IndexError:
        print("merge_boxes: 잘못된 데이터 형식입니다.", detections)
        return []

    # 박스 좌표 계산
    boxes = np.stack([
        x_centers - widths / 2,
        y_centers - heights / 2,
        x_centers + widths / 2,
        y_centers + heights / 2
    ], axis=1)
    
    # IoU 통합에서 신뢰도를 고려하지 않으므로 점수는 임의의 값 사용
    scores = np.ones(len(boxes), dtype=np.float32)
    
    # NMS 실행
    try:
        indices = torch.ops.torchvision.nms(
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(scores, dtype=torch.float32),
            iou_threshold
        )
    except Exception as e:
        print("merge_boxes: NMS 실행 중 오류 발생:", e)
        return []

    # NMS 통과한 결과만 반환
    merged_detections = detections[indices.numpy()]
    return merged_detections.tolist()

def detect_and_save(image_path, model, output_dir, img_size=640, conf_thres=0.1, iou_thres=0.45, repeat=10):
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return

    # Letterbox로 이미지 비율 유지하며 리사이즈
    img, ratio, pad = letterbox(image, img_size, stride=model.stride, auto=True)
    img_height, img_width = image.shape[:2]  # 원본 이미지 크기

    # 이미지 정규화 및 텐서 변환
    img = torch.from_numpy(img).float() / 255.0  # [0, 255] -> [0, 1]
    img = img.permute(2, 0, 1).unsqueeze(0)  # [HWC] -> [NCHW]

    results_list = []
    for _ in range(repeat):
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        detections = []  # 빈 리스트로 초기화
        if pred[0] is not None:
            for det in pred[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                
                # 패딩과 비율을 고려하여 원본 이미지 좌표로 변환
                x1 = (x1 - pad[0]) / ratio[0]
                y1 = (y1 - pad[1]) / ratio[1]
                x2 = (x2 - pad[0]) / ratio[0]
                y2 = (y2 - pad[1]) / ratio[1]

                # 정규화된 좌표 계산 (원본 이미지 크기로 나누기)
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # 정규화된 결과 추가
                detections.append([cls, x_center, y_center, width, height])
            results_list.extend(detections)

    if not results_list:
        print(f"탐지 결과가 없습니다: {image_path}")
        return

    # 기존 텍스트 파일 불러오기
    label_name = Path(image_path).stem
    label_path = os.path.join(output_dir, f"{label_name}.txt")
    existing_labels = load_existing_labels(label_path)

    # 새로운 결과 중 기존에 겹치지 않는 것만 추가
    new_results = []
    for detection in results_list:
        _, x_center, y_center, width, height = detection
        is_unique = True
        for existing in existing_labels:
            _, ex_x_center, ex_y_center, ex_width, ex_height = existing
            iou = calculate_iou([x_center, y_center, width, height], [ex_x_center, ex_y_center, ex_width, ex_height])
            if iou > 0.5:
                is_unique = False
                break
        if is_unique:
            new_results.append(detection)

    # 최종 결과 병합
    final_results = existing_labels + new_results

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{label_name}.txt")

    with open(output_path, 'w') as f:
        for result in final_results:
            cls, x_center, y_center, width, height = result
            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"탐지 결과가 저장되었습니다: {output_path}")

def batch_detect_and_save(image_dir, model_path, output_dir, repeat=10):
    model = load_model(model_path)

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_dir, image_file)
            print(f"처리 중: {image_path}")
            try:
                detect_and_save(image_path, model, output_dir, repeat=repeat)
            except Exception as e:
                print(f"오류 발생: {e}, 파일 건너뜀: {image_file}")

# 사용 예시
image_dir = CURRENT_DIR / "datasets/RGB_lights/images" # "autoData/forAuto-image/"  # 입력 이미지 디렉터리
model_path = CURRENT_DIR / "yolov5/runs/train/exp11/weights/best.pt"  # 모델 경로
output_dir = CURRENT_DIR / "datasets/RGB_lights/labels" # "autoData/forAuto-output/"  # 결과 저장 디렉터리

batch_detect_and_save(image_dir, model_path, output_dir, repeat=10)
