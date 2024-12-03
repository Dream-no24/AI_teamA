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

process_txt_files('yolov5/traffic_images/train/labels')