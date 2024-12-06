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
from utils.general import check_img_size, non_max_suppression
from utils.augmentations import letterbox  # Letterbox 추가

def load_model(model_path):
    device = select_device("")  # GPU/CPU 자동 선택
    model = DetectMultiBackend(model_path, device=device)  # 모델 로드
    model.warmup()  # 워밍업 (옵션)
    return model

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

    results_array = np.array(results_list)
    avg_results = results_array.mean(axis=0)

    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}.txt")

    with open(output_path, 'w') as f:
        f.write(f"{int(avg_results[0])} {avg_results[1]:.6f} {avg_results[2]:.6f} {avg_results[3]:.6f} {avg_results[4]:.6f}")
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
image_dir = CURRENT_DIR / "autoData/forAuto-image/"  # 입력 이미지 디렉터리
model_path = CURRENT_DIR / "yolov5/runs/train/exp5/weights/best.pt"  # 모델 경로
output_dir = CURRENT_DIR / "autoData/forAuto-output/"  # 결과 저장 디렉터리

batch_detect_and_save(image_dir, model_path, output_dir, repeat=10)
