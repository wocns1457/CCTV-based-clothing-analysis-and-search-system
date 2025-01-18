import os
import random
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from cctv_utils import resize_keeping_aspect_ratio, add_padding_random_position


class CCTV_DataSet(Dataset):
    def __init__(self, data, label, data_dir, data_name=None, processor=None, transform=None):

        self.data = data
        self.label = label
        self.text = [self.generate_prompt(value) for value in label]

        self.dir = data_dir
        self.data_name = data_name

        self.processor = processor
        self.transform = transform

        self.img_size = (processor.image_processor.size['width'], processor.image_processor.size['height'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]

        if self.data_name == 'peta':
            image_split = image.split('_')
            dataset, _, image = image_split[0], image_split[1], '_'.join(image_split[1:])
            image = f'{dataset}/archive/{image}'

        image = Image.open(os.path.join(self.dir, image))

        if random.random() > 0.5:
            image = resize_keeping_aspect_ratio(image, self.img_size, scale_factor=1.5)
            image = add_padding_random_position(image, self.img_size)

        if self.transform:
            image = self.transform(image)

        text = self.text[index]

        data = self.processor(images=image, text=text, padding='max_length', max_length=40, truncation=True, return_tensors="pt")

        data['pixel_values'] = data['pixel_values'].squeeze(0)
        data['input_ids'] = data['input_ids'].squeeze(0)
        data['attention_mask'] = data['attention_mask'].squeeze(0)

        return data

    def regenerate_prompt(self):
        self.text = [self.generate_prompt(value) for value in self.label]

    def generate_prompt(self, label):
        # 성별
        sex = label['sex']

        # 머리 정보
        hair_type = label['hair Type']   # 대머리일 때 조건
        hair_color = ' and '.join(label['hair Color'])

        # 상의 정보
        upper_body_other = label['upper Body Other']
        upper_body_style = label['upper Body Style']
        upper_body_sleeve = label['upper Body Sleeve']
        upper_body_type = label['upper Body Type']
        upper_body_point = label['upper Body Point']
        upper_body_color = ' and '.join(label['upper Body Color'])

        # 하의 정보
        lower_body_style = label['lower Body Style']
        lower_body_type = label['lower Body Type']
        lower_body_point = label['lower Body Point']
        lower_body_color = ' and '.join(label['lower Body Color'])

        # 신발 정보
        footwear_type = label['footwear Type']
        footwear_color = ' and '.join(label['footwear Color'])

        # 공통 부분
        hair_part = f"with {hair_color} {hair_type}," if hair_color or hair_type else ""

        upper_body_part = f"{upper_body_style} {upper_body_color} {upper_body_sleeve}"
        if upper_body_other == 'other' and upper_body_type and upper_body_type != 'plaid':
            upper_body_part += f" and {upper_body_type}"
        elif upper_body_type:
            upper_body_part += f" {upper_body_type}"
        if upper_body_point:
            upper_body_part += f" with {upper_body_point}"

        lower_body_part = f"{lower_body_style} {lower_body_color} {lower_body_type}"
        if not lower_body_style and not lower_body_type and not lower_body_color:
            lower_body_part = ''
        elif not lower_body_type:
            lower_body_type = 'pants'
            lower_body_part = f"{lower_body_style} {lower_body_color} {lower_body_type}"
        else:
            lower_body_part = f"{lower_body_style} {lower_body_color} {lower_body_type}"
        if lower_body_part and lower_body_point:
            lower_body_part += f" with {lower_body_point}"

        if not footwear_color and not footwear_type:
            footwear_part = ''
        elif footwear_color and not footwear_type:
            footwear_part = f"and {footwear_color} shoes"
        else:
            footwear_part = f"and {footwear_color} {footwear_type}"

        style = random.choice(['style1', 'style2', 'style3', 'style4', 'style5'])

        if style == 'style1':
            prompt = f"A {sex} {hair_part} wearing a {upper_body_part}, {lower_body_part} {footwear_part}."
        elif style == 'style2':
            prompt = f"{sex.capitalize()} in {upper_body_color} {upper_body_sleeve} and {lower_body_color} {lower_body_type}."
        elif style == 'style3':
            prompt = f"{sex.capitalize()} is wearing a {upper_body_part}. The outfit is completed with {lower_body_part} {footwear_part}."
        elif style == 'style4':
            footwear_part = f', paired with {footwear_color} {footwear_type}' if footwear_part else footwear_part
            prompt = f"{sex.capitalize()} {hair_part} dressed in a {upper_body_style} {upper_body_sleeve} and {lower_body_color} {lower_body_type} {footwear_part}."
        elif style == 'style5':
            prompt = f"{sex.capitalize()} {hair_part} wearing a {upper_body_color} {upper_body_sleeve} {upper_body_type}. {sex.capitalize()} also has {lower_body_color} {lower_body_type} {footwear_part}."

        return ' '.join(prompt.split()).replace(' ,', ',').replace(' .', '.')


class DeepFashionDataset(Dataset):
    def __init__(self, data, label, data_dir, processor, transform=None):

        self.data = data
        self.text = label

        self.dir = data_dir
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        text = self.text[index]

        image = Image.open(os.path.join(self.dir, image))
        text = self.generate_prompt(text)

        if self.transform:
            image = self.transform(image)

        data = self.processor(images=image, text=text, padding='max_length', max_length=40, truncation=True, return_tensors="pt")

        data['pixel_values'] = data['pixel_values'].squeeze(0)
        data['input_ids'] = data['input_ids'].squeeze(0)
        data['attention_mask'] = data['attention_mask'].squeeze(0)

        return data

    def generate_prompt(self, text):
        prompt = [text[0]]

        # 추가 텍스트 선택
        if len(text) > 1 and random.random() > 0.5:
            addition_texts = random.sample(text[1:], random.randint(1, len(text[1:])))

            for i, addition in enumerate(addition_texts):
                if i == 0:
                    prompt.append(", " + addition.lower())
                else:
                    prompt.append(" and " + addition.lower())

        prompt = ''.join(prompt)
        prompt = prompt if prompt[-1] == '.' else prompt + '.'

        return prompt
