# This file includes portions of YOLOv7's detect.py

import os
import sys
# yolov7 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov7'))

import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from yolov7.utils.datasets import letterbox
from yolov7.utils.general import  non_max_suppression

from cctv_utils import crop_image


def save_test_plot(image, text, dir='./test_result'):
    if not os.path.exists(dir):
        os.makedirs(dir)

    fig, ax = plt.subplots(2, 1, figsize=(8, 5))

    ax[0].imshow(image)
    ax[0].axis('off')

    ax[1].text(0.5, 0.7, text, ha='center', va='center', wrap=True, fontsize=12)
    ax[1].axis('off')
    plt.tight_layout()

    save_name = text.split(' ')[0]
    plt.savefig(f'{dir}/{save_name}.jpeg')
    plt.show()

# 영상이나 이미지 데이터를 프레임 단위로 처리하여 dictionary로 반환
def make_frame_dict(data_path, data_type, stride, imgsz=640, sampling_rate=None):
    frame_dict = {}
    frame_idx = 0
    if data_type == 'video':
        cap = cv2.VideoCapture(data_path)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps) if sampling_rate is None else sampling_rate

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                frame = letterbox(frame, imgsz, stride=stride)[0]
                frame = frame[:, :, ::-1].transpose(2, 0, 1)
                frame = np.ascontiguousarray(frame)
                frame_dict[str(frame_idx)] = frame
            frame_idx += 1
        cap.release()

    elif data_type == 'image':
        fps = 1
        img = cv2.imread(data_path)
        img = letterbox(img, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        frame_dict[str(frame_idx)] = img

    return frame_dict, fps

# YOLO 모델을 사용하여 프레임에서 바운딩 박스(Bounding Box)를 예측
def pred_bbox(model, device, half, frame_dict, imgsz=640, conf_thres=0.25, iou_thres=0.45, frame_bs=8):
    bbox_dict = {}
    frame_values = list(frame_dict.values())

    if device != 'cpu':
        model(torch.zeros(frame_bs, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = frame_bs

    for i in range(0, len(frame_values), frame_bs):
        frame_idx_list = list(frame_dict.keys())[i: min(len(frame_values), i+frame_bs)]
        frames = frame_values[i: min(len(frame_values), i+frame_bs)]
        frames = [torch.from_numpy(frame) for frame in frames]
        frames = torch.stack(frames, dim=0).to(device)
        frames = frames.half() if half else frames.float()
        frames /= 255.0

        # Warmup
        if i == 0 and device != 'cpu' and (old_img_b != frames.shape[0] or old_img_h != frames.shape[2] or old_img_w != frames.shape[3]):
            old_img_b = frames.shape[0]
            old_img_h = frames.shape[2]
            old_img_w = frames.shape[3]
            for i in range(3):
                model(frames, augment=False)[0]

        with torch.no_grad():
            pred = model(frames, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, agnostic=False)

        for j in range(len(pred)):
            frame_idx = frame_idx_list[j]
            pred_ = pred[j][:, :4].int().cpu().numpy()
            bbox_dict[frame_idx] = pred_.tolist() if len(pred[j]) else []

    return bbox_dict


def pre_caption(caption):
    caption = re.sub(
        r"([!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    return caption

# BLIP 모델을 사용하여 예측된 바운딩 박스 내 이미지를 기반으로 캡션을 생성
def generate_caption(model, device, processor, frame_dict, frame_num, bbox):
    imgs = []
    for i in range(len(bbox)):
        img = frame_dict[frame_num[i]].transpose(1, 2, 0)
        imgs.append(crop_image(img, bbox[i], crop_size=384))

    data = processor(images=imgs, return_tensors="pt").to(device)

    caption = model.generate(pixel_values=data['pixel_values'], max_new_tokens=40)
    caption = processor.batch_decode(caption, skip_special_tokens=True)

    return caption

# BLIP 모델을 사용하여 텍스트-이미지 간 유사성을 계산하고, 상위 K개의 결과를 반환
@torch.no_grad()
def compute_similarity(model, frame_dict, bbox_dict, prompt, processor, device, bs=32, num_top_k=10):
    imgs = []
    img_to_frame_map = []
    image_embeds = []

    prompt = pre_caption(prompt)
    prompt = processor(text=prompt, padding='max_length', max_length=30, truncation=True, return_tensors="pt").to(device)

    outputs = model.text_encoder(input_ids=prompt['input_ids'],
                                 attention_mask=prompt['attention_mask'])[0]

    text_embeds = F.normalize(model.text_proj(outputs[:, 0, :]), dim=-1)

    for idx, (frame_idx, bbox) in enumerate(bbox_dict.items()):
        if bbox:
            img = frame_dict[frame_idx].transpose(1, 2, 0)
            imgs.extend([crop_image(img, point, crop_size=384) for point in bbox])
            img_to_frame_map.extend([frame_idx] * len(bbox))

        if imgs and (len(imgs) >= bs or idx == len(bbox_dict.items())-1):
            data = processor(images=imgs, return_tensors="pt").to(device)

            outputs = model.vision_model(pixel_values=data['pixel_values'])[0]
            outputs = F.normalize(model.vision_proj(outputs[:, 0, :]), dim=-1)

            image_embeds.append(outputs)

            imgs.clear()

    image_embeds = torch.cat(image_embeds,dim=0)

    sims_matrix = text_embeds @ image_embeds.t()
    sims_matrix  = (sims_matrix + 1) / 2

    if not num_top_k==1:
        test_k = num_top_k * 2 if len(image_embeds) >= num_top_k * 2 else num_top_k
        topk_sim, topk_idx = sims_matrix.topk(k=test_k, dim=1)
        score_matrix_t2i = torch.full_like(sims_matrix, -100.0, device=sims_matrix.device)

        imgs = []
        for idx in topk_idx[0]:
            frame_idx = img_to_frame_map[idx]
            frame_idx_indices = [i for i, x in enumerate(img_to_frame_map) if x == frame_idx]
            bbox = bbox_dict[frame_idx][frame_idx_indices.index(idx)]
            img = frame_dict[frame_idx].transpose(1, 2, 0)
            imgs.append(crop_image(img, bbox, crop_size=384))

        data = processor(images=imgs, return_tensors="pt").to(device)

        output = model(pixel_values=data['pixel_values'],
                      input_ids=prompt['input_ids'].repeat(test_k, 1),
                      attention_mask=prompt['attention_mask'].repeat(test_k, 1),
                      return_loss=False,
                      task='itm'
                      )

        score = output['itm']['outputs'][:, 1]
        score = (F.normalize(score.unsqueeze(0)) + 1) / 2

        score_matrix_t2i[:, topk_idx] = (score + topk_sim) / 2

        topk_sim, topk_idx = score_matrix_t2i.topk(k=num_top_k, dim=1)
    else:
        topk_sim, topk_idx = sims_matrix.topk(k=num_top_k, dim=1)

    topk_sim = [round(score, 4) for score in topk_sim.squeeze(0).tolist()]
    topk_idx = topk_idx.squeeze(0).tolist()

    # topk_sim_frame_num = [img_to_frame_map[idx.item()] for idx in topk_idx.squeeze(0)]
    topk_sim_frame_num = [img_to_frame_map[idx] for idx in topk_idx]

    return topk_sim, topk_idx, topk_sim_frame_num
