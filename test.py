import os
import sys
import argparse

# yolov7 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov7'))

from PIL import Image
from itertools import chain

import cv2
import numpy as np
import torch

from transformers import AutoProcessor

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size

from test_utils import make_frame_dict, pred_bbox, compute_similarity, generate_caption, save_test_plot
from model import CCTV_Analysis_Model


def inference(args):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load yolo model
    yolo_model = attempt_load(args.yolo_weights, map_location=device)
    yolo_model.eval()

    if half:
        yolo_model.half()

    stride = int(yolo_model.stride.max())
    imgsz = check_img_size(640, s=stride)

    # Load BLIP model
    blip_model = CCTV_Analysis_Model(load_pretrained_model=False)
    blip_model.load_state_dict(torch.load(args.blip_weights, weights_only=True, map_location=device)['model'])
    blip_model.to(device)
    blip_model.eval()

    vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
    data_type = 'video' if args.file_path.split('.')[-1].lower() in vid_formats else 'image'

    frame_bs = 8 if data_type == 'video' else 1
    blip_bs = 50 if device.type != 'cpu' else 1

    frame_dict, fps = make_frame_dict(args.file_path, data_type, imgsz=imgsz, stride=stride, sampling_rate=args.sampling_rate)

    print("generated {} frames".format(len(frame_dict)))

    # 예측 수행
    print("Detecting people...")
    bbox_dict = pred_bbox(yolo_model, device, half, frame_dict, imgsz=imgsz, conf_thres=args.conf_thres, iou_thres=0.45, frame_bs=frame_bs)

    if all(not value for value in bbox_dict.values()):
        return "사람이 검출되지 않았습니다."

    # 유사도 계산
    count_box = sum(len(bbox) for bbox in bbox_dict.values())

    if data_type == 'video':
        num_top_k = args.num_top_k if count_box >= args.num_top_k else count_box
    else:
        num_top_k = 1

    print("Calculating similarity...")
    topk_sim, topk_idx, topk_frame_num = compute_similarity(blip_model, frame_dict, bbox_dict, args.prompt, processor, device, bs=blip_bs, num_top_k=num_top_k)

    bbox_list = list(chain.from_iterable(bbox_dict.values()))
    topk_bbox = [bbox_list[i] for i in topk_idx]

    print("Generating captions...")
    caption = generate_caption(blip_model, device, processor, frame_dict, topk_frame_num, topk_bbox)

    # 결과 반환
    for i, frame_num in enumerate(topk_frame_num):
        if topk_sim[i] >= args.sim_thres:
            frame = frame_dict[str(frame_num)].transpose(1, 2, 0)

            pil_img = Image.fromarray(frame)

            seconds = int(frame_num) / fps
            minutes = int(seconds // 60)
            seconds = seconds % 60

            result = f'Top{i+1} similarity: {topk_sim[i]}, time: {minutes} minutes {seconds:.2f} seconds \n{caption[i]}'

            save_test_plot(pil_img, result, dir=args.result_dir)

    print("File processed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='path of image or video')
    parser.add_argument('--prompt', type=str, help='prompt for calculating similarity with image')

    parser.add_argument('--yolo_weights', type=str, default='./weights/yolov7.pt')
    parser.add_argument('--blip_weights', type=str, default='./weights/blip.pth')
    parser.add_argument('--result_dir', type=str, default='./test_result')

    parser.add_argument('--conf_thres', type=float, default=0.25, help='detection confidence threshold')
    parser.add_argument('--sim_thres', type=float, default=0.50, help='text-image similarity threshold')
    parser.add_argument('--sampling_rate', type=int, default=30, help='video sampling rate')
    parser.add_argument('--num_top_k', type=int, default=1, help='top k text-image similarity')

    args = parser.parse_args()
    print(args)

    inference(args)