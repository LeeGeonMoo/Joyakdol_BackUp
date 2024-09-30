import sys
sys.path.append("/home/ubuntu/projects/myapi/joyakdol_230715/yolov5_master/")

import torch
from PIL import Image
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression
import numpy as np

# YOLOv5 모델 로드
weights = '/home/ubuntu/projects/myapi/joyakdol_230715/best.pt' # 안바뀜 얘는
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights)

# 얘부터 이제 함수 영역. 파라미터 필요.

def check_medicine_photo(num):
    # 이미지 경로 설정
    image_path = f"/home/ubuntu/projects/myapi/image/{num}.jpg" # 경로 수정은 항상 필요.

    # 이미지 로드
    image = Image.open(image_path)

    # 이미지 전처리
    img_size = 416
    image = image.resize((img_size, img_size))

    # 이미지를 Tensor로 변환
    #image_tensor = torch.tensor(image, dtype=torch.float32).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    image_array = np.array(image, dtype=np.float32)  # 이미지를 NumPy 배열로 변환 / 하도 앞에게 안돼서.. 챗지피티이용!
    image_tensor = torch.tensor(image_array).unsqueeze(0).permute(0, 3, 1, 2) / 255.0

    # 모델 추론
    pred = model(image_tensor.to(device))[0]
    pred = non_max_suppression(pred, conf_thres=0.2)
    
    # 결과값 프린트 
    
    #print(int(pred[0][0][5])) # 그래도 혹시 모르니 try / error 필요할듯. 오류 발생하면 큰일나니까.
    try :
        return int(pred[0][0][5])
    except :
        return -1            
    # 오류나면 -1번 리턴하도록.
