import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (Profile, check_file, check_img_size,check_requirements, 
                           non_max_suppression, scale_boxes,  xyxy2xywh)

from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def check_medicine_photo(num):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

# 오류나서 이거 추가해줌
    weights='/home/ubuntu/projects/myapi/joyakdol_230715/best_230804.pt'
# model path or triton URL
# weights / source 변경해놓음.
    data=ROOT / 'data/coco128.yaml' # dataset.yaml path
    imgsz=(416, 416)  # inference size (height, width)
    conf_thres=0.2  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    save_txt=True  # save results to *.txt
    save_conf=True  # save confidences in --save-txt labels
    # nosave를 True로 -> 사진 파일 저장 안되게끔. 빙고!
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    half=False  # use FP16 half-precision inference
    dnn=False  # use OpenCV DNN for ONNX inference
    vid_stride=1  # video frame-rate stride
    
    source=f"/home/ubuntu/projects/myapi/image/{num}.jpg"  # file/dir/URL/glob/screen/0(webcam)

    med_list = []
    conf_list = []
        
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    if is_url and is_file:
        source = check_file(source)  # download

    # Load model -> 모델 로드는 필요하지.
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # 무조건 이미지니까.
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    # 대충 모델 돌리는거겠지 이게.
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Write results
                
                # 파일 적는 부분. 여기의 변수들만 잘 추린다면!
                
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh / 박스 위치 정규화.
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        
                        print(line)
			#230817 추가
                        med_list.append(line)

                for x in med_list:
                    conf_list.append(x[5])
                if max(conf_list) >= 0.8:
                    med_id = med_list[conf_list.index(max(conf_list))][0]
                    return int(med_id)
                else :
                    return -1
            else :
                return -1




#check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
#실행할때 왜인지는 모르겠지만 이거 필요함.

#print(check_medicine_photo(1))
