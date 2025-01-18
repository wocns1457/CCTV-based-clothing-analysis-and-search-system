import os
import io
import sys

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov7'))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import base64
import tempfile
import json

from PIL import Image
from itertools import chain

from flask import Flask, request, Response, stream_with_context

import torch

from transformers import AutoProcessor

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size

from test_utils import make_frame_dict, pred_bbox, compute_similarity, generate_caption
from model import CCTV_Analysis_Model


app = Flask(__name__)

@app.route('/')
def ping():
    return 'ping test'

@app.route('/predict_video', methods=['POST'])
def predict_video():
    def generate():
        try:
            print("Request comes in !!")

            prompt = request.form.get('prompt')
            data_type = request.form.get('type')
            use_sample = request.form.get('use_sample')
            file_name = request.form.get('file_name')
            conf_thres = float(request.form.get('conf_thres'))
            sim_thres = float(request.form.get('sim_thres'))
            num_top_k = int(request.form.get('num_top_k'))
            sampling_rate = int(request.form.get('sampling_rate'))

            frame_bs = 8 if data_type == 'video' else 1
            blip_bs = 50 if device.type != 'cpu' else 1

            yield 'data: {}\n\n'.format(json.dumps({"status": "received parameters", "prompt": prompt}))

            if use_sample == 'No' and 'file' not in request.files:
                yield 'data: {}\n\n'.format(json.dumps({"error": "No file provided"}))
                return
            if 'prompt' not in request.form:
                yield 'data: {}\n\n'.format(json.dumps({"error": "No prompt provided"}))
                return

            if use_sample == 'No':
                file = request.files['file']
                file_bytes = file.read()
                suffix = '.mp4' if data_type == 'video' else '.jpeg'
                with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as temp_file:
                    temp_file.write(file_bytes)
                    temp_file_path = temp_file.name
                    frame_dict, fps = make_frame_dict(temp_file_path, data_type, imgsz=imgsz, stride=stride, sampling_rate=sampling_rate)
            else:
                file_path = os.path.join('../sample', file_name)
                frame_dict, fps = make_frame_dict(file_path, data_type, imgsz=imgsz, stride=stride, sampling_rate=sampling_rate)

            # 예측 수행
            yield 'data: {}\n\n'.format(json.dumps({"status": "Detecting people..."}))
            print("Detecting people...")
            bbox_dict = pred_bbox(yolo_model, device, half, frame_dict, imgsz=imgsz, conf_thres=conf_thres, iou_thres=0.45, frame_bs=frame_bs)

            if all(not value for value in bbox_dict.values()):
                yield 'data: {}\n\n'.format(json.dumps({"error": "사람이 검출되지 않았습니다."}))
                return

            # 유사도 계산
            count_box = sum(len(bbox) for bbox in bbox_dict.values())
            num_top_k = num_top_k if count_box >= num_top_k else count_box

            yield 'data: {}\n\n'.format(json.dumps({"status": "Calculating similarity..."}))
            print("Calculating similarity...")
            topk_sim, topk_idx, topk_frame_num = compute_similarity(blip_model, frame_dict, bbox_dict, prompt, processor, device, bs=blip_bs, num_top_k=num_top_k)

            bbox_list = list(chain.from_iterable(bbox_dict.values()))
            topk_bbox = [bbox_list[i] for i in topk_idx]

            yield 'data: {}\n\n'.format(json.dumps({"status": "Generating captions..."}))
            print("Generating captions...")
            caption = generate_caption(blip_model, device, processor, frame_dict, topk_frame_num, topk_bbox)

            # 결과 반환
            results = {'result':[]}
            for i, frame_num in enumerate(topk_frame_num):
                if topk_sim[i] >= sim_thres:
                    frame = frame_dict[str(frame_num)].transpose(1, 2, 0)

                    pil_img = Image.fromarray(frame)
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format='JPEG')
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    seconds = int(frame_num) / fps

                    result = {'image': img_str, 'caption': caption[i], 'similarity': topk_sim[i], 'seconds': seconds}
                    results['result'].append(result)
                    print( caption[i], topk_sim[i], seconds)

            yield 'data: {}\n\n'.format(json.dumps(results))

            yield 'data: {}\n\n'.format(json.dumps({"status": "completed"}))
            print("File processed successfully!")

        except Exception as e:
            yield 'data: {}\n\n'.format(json.dumps({"error": str(e)}))

    return Response(stream_with_context(generate()))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'

    # Load yolo model
    yolo_weights = '../weights/yolov7_training.pt'
    yolo_model = attempt_load(yolo_weights, map_location=device)
    yolo_model.eval()

    if half:
        yolo_model.half()

    stride = int(yolo_model.stride.max())
    imgsz = check_img_size(640, s=stride)

    # Load blip model
    blip_weights = '../weights/blip.pth'
    blip_model = CCTV_Analysis_Model(load_pretrained_model=False)
    blip_model.load_state_dict(torch.load(blip_weights, weights_only=True, map_location=device)['model'])
    blip_model.to(device)
    blip_model.eval()

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    app.run(host='0.0.0.0', port=5000)
