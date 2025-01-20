import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import random
import json
import argparse
import cv2
from PIL import Image

from cctv_utils import crop_image
from data.data_utils import check_available_data, modify_word


def check_other_person(id, frame, categories):
    for cate in categories:
        if cate['id'] != id and cate['cam_in'] <= frame <= cate['cam_out']:
              return False
    return True

def modify_annotation(label, config):
    label_template_copy = config['template'].copy()
    label_template_copy['sex'] = modify_word(label['gender'], config)
    label_template_copy['upper Body Sleeve'] = label['top_type'].replace('_', ' ')
    label_template_copy['upper Body Color'] = [label['top_color']]
    label_template_copy['lower Body Type'] = label['bottom_type'].replace('_', ' ')
    label_template_copy['lower Body Color'] = [label['bottom_color']]
    return label_template_copy

def make_annotation(label):
    annotations = {}
    for data in label['annotations']:
        data['bbox_extent'] = (data['bbox'][2] - data['bbox'][0]) * (data['bbox'][3] - data['bbox'][1])
        if data['id'] in annotations:
            annotations[data['id']].append(data)
        else:
            annotations[data['id']] = [data]

    for person in annotations:
        annotations[person] = sorted(annotations[person], key=lambda x: x['bbox_extent'], reverse=True)
        choice_id = annotations[person][0]['id']
        choice_frame = annotations[person][0]['frame']
        max_extent = annotations[person][0]['bbox_extent']
        use_full_size = False

        for info in annotations[person]:
            if check_other_person(info['id'], info['frame'], label['categories']) and max_extent*0.7 < info['bbox_extent']:
                choice_frame = info['frame']
                use_full_size = True
                break

        annotations[choice_id] = [person for person in annotations[choice_id] if person['frame'] == choice_frame][0]
        annotations[choice_id]['gender'] = [person['gender'] for person in label['categories'] if person['id'] == choice_id][0]
        annotations[choice_id]['use_full_size'] = use_full_size

    return annotations

def get_frames(video_path, ann):
    frames = []
    video_name = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    for person in list(ann.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, ann[person]['frame'])
        _, image = cap.read()

        if ann[person]['use_full_size']:
            # image = cv2.resize(image, (384, 384))
            image = crop_image(image, ann[person]['bbox'], crop_size=384)
        else:
            bbox = ann[person]['bbox']
            image = image[int(bbox[1]) :int(bbox[3]), int(bbox[0]):int(bbox[2])]

        if check_available_data(Image.fromarray(image)):
            frames.append((f'{video_name}_{person}.png', image))
        else:
            del ann[person]
    cap.release()
    return frames, ann
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='./ye-ma')
    parser.add_argument('--prompt_config_path', type=str, default='./prompt_config.json')
    parser.add_argument('--save_dir', type=str, default='./AI_HUB_CCTV')
    args = parser.parse_args()

    video_dir = args.video_dir
    prompt_config_path = args.prompt_config_path
    save_dir = args.save_dir

    prompt_config = json.load(open(prompt_config_path, "r"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_file_list = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".json"):
                label_file_list.append(os.path.join(root, file))

    # label_file_list = random.sample(label_file_list, 300)

    all_annotations = {}

    for label_file in label_file_list:
        video_path = label_file.replace('.json', '.mp4')
        label = json.load(open(label_file, "r"))
        annotations = make_annotation(label)
        frames, annotations = get_frames(video_path, annotations)

        for i, ann in enumerate(annotations.values()):
            frame_name, frame = frames[i][0], frames[i][1]
            all_annotations[frame_name] = modify_annotation(ann, prompt_config)
            cv2.imwrite(os.path.join(save_dir, frame_name), frame)

    with open(os.path.join(save_dir, "label_ye-ma.json"), "w") as f:
        f.write(json.dumps(all_annotations, indent=4) + "\n")
