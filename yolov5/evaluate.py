import torch
from pathlib import Path
from val import run  # val.py의 run 함수 사용
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix
import seaborn as sns

def evaluate_model(weights_path, data_path, img_size=640, batch_size=1, device=None):
    try:
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 절대 경로로 변환
        weights_path = str(Path(weights_path).absolute())
        data_path = str(Path(data_path).absolute())

        print(f"Evaluating model with weights: {weights_path}")
        print(f"Using dataset config: {data_path}")

        # val.py의 run 함수 호출
        results = run(
            weights=weights_path,  # 모델 가중치
            data=data_path,        # 데이터셋 구성 파일
            imgsz=img_size,        # 이미지 크기
            batch_size=batch_size, # 배치 크기
            device=device,         # 평가에 사용할 장치
            conf_thres=0.001,      # Confidence Threshold
            iou_thres=0.6,         # IoU Threshold
            save_json=False,       # COCO API 평가 JSON 저장 비활성화
            save_txt=False,        # 결과 텍스트 저장 비활성화
            verbose=False          # 세부 정보 출력 비활성화
        )

        # results[0][0]을 통해 precision에 접근
        precision = results[0][0]  # 첫 번째 값인 precision
        recall = results[0][1]     # 두 번째 값은 recall
        mAP_0_5 = results[0][2]   # mAP@0.5
        mAP_0_5_95 = results[0][3]  # mAP@0.5:0.95
      

        print("\nEvaluation Results:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"mAP@0.5: {mAP_0_5:.4f}")
        print(f"mAP@0.5:0.95: {mAP_0_5_95:.4f}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 평가에 사용할 가중치 파일 및 데이터셋 구성 파일
    weights_file = "runs/train/exp/weights/best.pt"  # 모델 가중치 경로
    data_file = "data/traffic.yaml"                 # 데이터셋 구성 파일 경로

    # 평가 수행
    evaluate_model(weights_path=weights_file, data_path=data_file)