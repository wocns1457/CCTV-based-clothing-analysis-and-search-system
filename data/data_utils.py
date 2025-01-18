import os
import re
import random
from PIL import Image

"""
Functions for processing PETA dataset
"""
def merge_dataset(data_list, dir):
    merge_data=[]
    for dataset in data_list:
        img_dir = f'{dir}/{dataset}/archive'
        data = sorted(os.listdir(img_dir))[:-1]
        data = [f'{dataset}_{img}' for img in data if check_available_data(f'{img_dir}/{img}')]
        merge_data.extend(data)
    return merge_data

def modify_word(word, config):
    if word in config['modifyWord']:
        word = config['modifyWord'][word]
        word = random.choice(word) if isinstance(word, list) else word
    return word

def make_template(label, config):
    label_template_copy = config['template'].copy()
    for attr in label[1:]:
        origin_attr = attr.replace(' ', '')
        attr_split = attr.split(' ')

        if origin_attr in config:
            attr_type = attr_split[0]if len(attr_split) == 2 else ''.join(attr_split[:2])
            item = re.findall(rf'(?<={attr_type}).+', origin_attr)[0]
            item = modify_word(re.sub(r'([A-Z])', r' \1', item).lower()[1:], config)
            label_template_copy[config[origin_attr]] = item
        else:
            attr_type = ' '.join(attr_split[:-1])
            color = attr_split[-1]
            color = color.lower() if color in config['colors'] else None

            if color is not None:
                if label_template_copy[f'{attr_type} Color']:
                    label_template_copy[f'{attr_type} Color'].append(color)
                else:
                    label_template_copy[f'{attr_type} Color'] = [color]

    return label_template_copy

def get_template(data_list, dir, config):
    label_dict={}
    for dataset in data_list:
        label_dict[dataset] = {}
        labels = open(f'{dir}/{dataset}/archive/Label.txt', 'r', encoding='utf8').readlines()

        for label in labels:
            label = [re.sub(r'([A-Z])', r' \1', attr) for attr in label.split()]
            label_dict[dataset][label[0]] = make_template(label, config)

        label_dict[dataset] = dict(sorted(label_dict[dataset].items()))

    return label_dict
    
def count_empty_value(label_template):
    return sum(1 for value in label_template.values() if value == '' or value[0]=='')

def check_available_data(img=None, label=None, size=(80, 160), num_missing_value=5, return_invalid_data=False):
    if img:
        img = img if isinstance(img, Image.Image) else Image.open(img)
        return False if size[0] > img.size[0] or size[1] > img.size[1] else True

    if label and not return_invalid_data:
        return False if count_empty_value(label) > num_missing_value else True

    elif label and return_invalid_data:
        invalid_data = []
        for dataset in label:
            for id, label_template in label[dataset].items():
                if count_empty_value(label_template) > num_missing_value:
                    invalid_data.append(f'{dataset}_{id}')
        return invalid_data

"""
Functions for processing DeepFashsion dataset
"""
def modify_text(text):
    item=['accessory', 'ring', 'wrist', 'neckwear', 'fabric']
    replacements = {"tank tank": "tank", "short-sleeve": "short sleeve", "long-sleeve": "long sleeve"}

    if text == '':
        return None
    else:
        sentences = text

        for old, new in replacements.items():
            sentences = sentences.replace(old, new)
        sentences = sentences.split('. ')
        sentences[-1] = sentences[-1].replace('.', '')
        sentences = [sentence.capitalize() for sentence in sentences if not any(word in sentence for word in item)]

        if len(sentences) <= 2:
            if not any(any(word in sentence for word in ['wears', 'wearing']) for sentence in sentences):
                return None

        change_idx = None
        for i, sentence in enumerate(sentences):
            if 'wears' in sentence or 'wearing' in sentence:
                change_idx = i
                break

        if change_idx is not None and change_idx != 0:
            sentences[0], sentences[change_idx] = sentences[change_idx], sentences[0]

        return sentences

def sampling_sex_data(label, target_size=10000):
    MEN_RATIO, WOMEN_RATIO = 0.4, 0.6
    include_full_women_data, other_women_data = [], []

    men_data = [k for k in label.keys() if k.startswith('MEN')]
    women_data = [k for k in label.keys() if k.startswith('WOMEN')]

    for data in women_data:
        if 'full' in data:
            include_full_women_data.append(data)
        else:
            other_women_data.append(data)

    if target_size <= len(label):
        if int(MEN_RATIO * target_size) <= len(men_data):
            men_data = random.sample(men_data, int(MEN_RATIO * target_size))
            women_target_size = target_size - len(men_data)
        else:
            women_target_size = target_size - len(men_data)

        if int(women_target_size * 0.7) <= len(include_full_women_data) and women_target_size <= len(include_full_women_data):
            include_full_women_data = random.sample(include_full_women_data, int(women_target_size*0.6))
            other_women_size = women_target_size - len(include_full_women_data)
            other_women_data = random.sample(other_women_data, other_women_size)
            women_data = include_full_women_data + other_women_data
        else:
            other_women_size = women_target_size - len(include_full_women_data)
            other_women_data = random.sample(other_women_data, other_women_size)
            women_data = include_full_women_data + other_women_data

        return {k: label[k] for k in set(men_data + women_data)}
    else:
        return label

def filtering_data(label, target_size=10000):
    fashion_item, remove_data = {}, []

    for data in list(label.keys()):
        text = modify_text(label[data])

        if text is None:
            del label[data]
        else:
            label[data] = text

            if not 'full' in data:
                id = data.split('-')
                id = f'{id[2]}-{id[3][:2]}'

                if id in fashion_item:
                    fashion_item[id].append(data)
                else:
                    fashion_item[id] = [data]

    for item_list in fashion_item.values():
        if len(item_list) >= 3:
            remove_data = random.sample(item_list, 2)

        elif len(item_list) > 1 and 'front' in item_list[0]:
            remove_data = random.sample(item_list[1:], 1)

        for _ in range(len(remove_data)):
            del label[remove_data.pop(0)]

    return sampling_sex_data(label, target_size=target_size)
