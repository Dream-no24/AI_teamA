import torch
import os

# 현재 파일과 동일한 디렉터리 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 절대 경로

# 기존 모델 파일 경로
model_path = os.path.join(current_dir, 'best.pt')

# 모델 로드
model = torch.load(model_path, map_location='cpu')

# state_dict 추출
state_dict = model['model'].state_dict() if 'model' in model else model.state_dict()

# 새로운 파일 경로 설정
new_model_path = os.path.join(current_dir, 'state_dict_model.pth')

# state_dict 저장
torch.save(state_dict, new_model_path)

print(f"State_dict가 저장되었습니다: {new_model_path}")
