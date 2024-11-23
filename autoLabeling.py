import sys
import os
import cv2
import torch
from pathlib import Path

# YOLOv5 경로를 autoLabeling.py 위치 기준으로 추가
CURRENT_DIR = Path(__file__).resolve().parent  # autoLabeling.py가 위치한 디렉터리
YOLOV5_PATH = CURRENT_DIR / "yolov5"  # yolov5 경로를 기준으로 설정
sys.path.insert(0, str(YOLOV5_PATH))

# YOLOv5 모듈 임포트
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression

def load_model(model_path):
    """
    YOLOv5 모델 로드
    Args:
        model_path (str): YOLOv5 학습된 모델(.pt) 경로
    Returns:
        모델 객체
    """
    device = select_device("")  # GPU/CPU 자동 선택
    model = DetectMultiBackend(model_path, device=device)  # 모델 로드
    model.warmup()  # 워밍업 (옵션)
    return model

def detect_and_save(image_path, model, output_dir, img_size=640, conf_thres=0.25, iou_thres=0.45):
    """
    이미지 탐지 및 결과 저장
    Args:
        image_path (str): 입력 이미지 경로
        model: 로드된 YOLO 모델
        output_dir (str): 결과 저장 디렉터리
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return

    # 이미지 크기 가져오기
    img_height, img_width = image.shape[:2]

    # 모델 추론 준비
    img_size = check_img_size(img_size, s=model.stride)
    img = cv2.resize(image, (img_size, img_size))
    img = torch.from_numpy(img).float() / 255.0  # [0, 255] -> [0, 1]
    img = img.permute(2, 0, 1).unsqueeze(0)  # [HWC] -> [NCHW]

    # 모델 추론
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # 탐지 결과 정리 및 정규화
    detections = []
    for det in pred[0]:  # 첫 번째 이미지의 결과만 가져옴
        x1, y1, x2, y2, conf, cls = det.tolist()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        # 정규화된 좌표 계산
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height

        # 클래스 번호와 정규화된 좌표 추가
        detections.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}.txt")
    with open(output_path, 'w') as f:
        f.write("\n".join(detections))
    print(f"탐지 결과가 저장되었습니다: {output_path}")

def batch_detect_and_save(image_dir, model_path, output_dir):
    """
    디렉터리 내 모든 이미지 탐지 및 결과 저장
    Args:
        image_dir (str): 이미지 디렉터리 경로
        model_path (str): YOLO 모델 경로
        output_dir (str): 결과 저장 디렉터리
    """
    # 모델 로드
    model = load_model(model_path)

    # 디렉터리 내 모든 이미지 처리
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_dir, image_file)
            print(f"처리 중: {image_path}")
            try:
                detect_and_save(image_path, model, output_dir)
            except Exception as e:
                print(f"오류 발생: {e}, 파일 건너뜀: {image_file}")

# 사용 예시
image_dir = CURRENT_DIR / "autoData/forAuto-image/"  # 입력 이미지 디렉터리
model_path = CURRENT_DIR / "yolov5/runs/train/exp/weights/best.pt"  # 모델 경로
output_dir = CURRENT_DIR / "autoData/forAuto-output/"  # 결과 저장 디렉터리

batch_detect_and_save(image_dir, model_path, output_dir)
