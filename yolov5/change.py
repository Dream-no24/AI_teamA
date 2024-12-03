import os

def process_txt_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            if lines:
                with open(file_path, 'w') as file:
                    file.write(lines[0])
                    
def print_files_with_more_than_two_lines(directory):
    # 디렉토리 내의 모든 파일을 순회
    for filename in os.listdir(directory):
        # 파일이 .txt로 끝나는지 확인
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                # 파일의 모든 줄을 읽음
                lines = file.readlines()
                # 두 줄 이상인 경우 파일 이름 출력
                if len(lines) >= 2:
                    print(filename)

#process_txt_files('yolov5/traffic_images/train/labels')
print_files_with_more_than_two_lines('yolov5/traffic_images/train/labels')
