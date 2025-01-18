import os
import json
import random
from sklearn.model_selection import train_test_split

from data.data_utils import merge_dataset, get_template, check_available_data, filtering_data

from data.dataset import CCTV_DataSet, DeepFashionDataset


def create_peta_dataset(data_dir, transform, processor, seed, use_data_size=None, train_size=0.7, config=None):
    peta_data, peta_text = merge_dataset(config['PETA_DATA_LIST'], data_dir), get_template(config['PETA_DATA_LIST'], data_dir, config)

    invalid_data = check_available_data(label=peta_text, num_missing_value=5, return_invalid_data=True)

    peta_data = [data for data in peta_data if not '_'.join(data.split('_')[:2]) in invalid_data]
    peta_data = random.sample(peta_data, use_data_size) if use_data_size is not None else peta_data

    train, val = train_test_split(peta_data, train_size=train_size, random_state=seed)
    val, test = train_test_split(val, train_size=0.65, random_state=seed)

    train_text = [peta_text[data.split('_')[0]][data.split('_')[1]] for data in train]
    val_text = [peta_text[data.split('_')[0]][data.split('_')[1]] for data in val]
    test_text = [peta_text[data.split('_')[0]][data.split('_')[1]] for data in test]

    train_dataset = CCTV_DataSet(data=train, label=train_text, data_dir=data_dir, data_name='peta', processor=processor, transform=transform)
    val_dataset = CCTV_DataSet(data=val, label=val_text, data_dir=data_dir, data_name='peta', processor=processor, transform=None)
    test_dataset = CCTV_DataSet(data=test, label=test_text, data_dir=data_dir, data_name='peta', processor=processor, transform=None)
    return train_dataset, val_dataset, test_dataset


def create_ai_hub_dataset(data_dir, transform, processor, seed, use_data_size=None, train_size=0.7, config=None):
    ai_hub_data = sorted(os.listdir(data_dir))
    ai_hub_img_list = ai_hub_data[:-1]
    ai_hub_label = json.load(open(os.path.join(data_dir, ai_hub_data[-1]), "r"))

    ai_hub_img_list = random.sample(ai_hub_img_list, use_data_size) if use_data_size is not None else ai_hub_img_list

    train, val = train_test_split(ai_hub_img_list, train_size=train_size, random_state=seed)
    val, test = train_test_split(val, train_size=0.65, random_state=seed)

    train_text = [ai_hub_label[data] for data in train]
    val_text = [ai_hub_label[data] for data in val]
    test_text = [ai_hub_label[data] for data in test]

    train_dataset = CCTV_DataSet(data=train, label=train_text, data_dir=data_dir, data_name=None, processor=processor, transform=transform)
    val_dataset = CCTV_DataSet(data=val, label=val_text, data_dir=data_dir, data_name=None, processor=processor, transform=None)
    test_dataset = CCTV_DataSet(data=test, label=test_text, data_dir=data_dir, data_name=None, processor=processor, transform=None)
    return train_dataset, val_dataset, test_dataset


def create_deepfashion_dataset(data_dir, transform, processor, seed, use_data_size=None, train_size=0.7):
    deepfashion_data = json.load(open(f'{data_dir}/captions.json', "r"))
    deepfashion_data = filtering_data(deepfashion_data, target_size=use_data_size)
    deepfashion_data_keys = list(deepfashion_data.keys())

    train, val = train_test_split(deepfashion_data_keys, train_size=train_size, random_state=seed)
    val, test = train_test_split(val, train_size=0.65, random_state=seed)

    train = {k: deepfashion_data[k] for k in train}
    val = {k: deepfashion_data[k] for k in val}
    test = {k: deepfashion_data[k] for k in test}

    train_dataset = DeepFashionDataset(data=list(train.keys()), label=list(train.values()), data_dir=data_dir, processor=processor, transform=transform)
    val_dataset = DeepFashionDataset(data=list(val.keys()), label=list(val.values()), data_dir=data_dir, processor=processor, transform=None)
    test_dataset = DeepFashionDataset(data=list(test.keys()), label=list(test.values()), data_dir=data_dir, processor=processor, transform=None)
    return train_dataset, val_dataset, test_dataset