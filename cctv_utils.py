import os
import re
import random
import json
import matplotlib.pyplot as plt
from PIL import Image


def resize_keeping_aspect_ratio(img, target_size, scale_factor=2.0):
    new_size = (int(img.size[0] * scale_factor if img.size[0]* scale_factor < target_size[0] else target_size[0]),
                int(img.size[1] * scale_factor if img.size[1]* scale_factor < target_size[1] else target_size[1]))

    return img.resize(new_size, Image.LANCZOS)

def add_padding_random_position(img, target_size):
    original_size = img.size

    background_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    new_img = Image.new("RGB", target_size, background_color)

    max_x_offset = target_size[0] - original_size[0]
    max_y_offset = target_size[1] - original_size[1]

    x_offset = random.randint(0, max_x_offset)
    y_offset = random.randint(0, max_y_offset)

    offset = (x_offset, y_offset)

    new_img.paste(img, offset)
    return new_img

def crop_image(image, bbox, crop_size=384):
    img_height, img_width = image.shape[:2]

    bbox = bbox = [int(i) for i in bbox]

    x_center = (bbox[0] + bbox[2]) // 2
    y_center = (bbox[1] + bbox[3]) // 2

    x_start = x_center - crop_size // 2
    y_start = y_center - crop_size // 2

    x_start = max(0, min(x_start, img_width - crop_size))
    y_start = max(0, min(y_start, img_height - crop_size))

    x_end = x_start + crop_size
    y_end = y_start + crop_size

    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image

def save_loss_plot(save_dir):
    with open(f'{save_dir}/train_log.json', 'r') as f:
        train_log = [json.loads(line) for line in f]

    with open(f'{save_dir}/eval_log.json', 'r') as f:
        eval_log = [json.loads(line) for line in f]

    train_log_dict = {}
    eval_log_dict = {}
    score_dict = {}

    train_log_dict['itm_loss'] = [loss_value for log in train_log for loss_value in log['itm_loss']]
    train_log_dict['itc_loss'] = [loss_value for log in train_log for loss_value in log['itc_loss']]
    train_log_dict['caption_loss'] = [loss_value for log in train_log for loss_value in log['caption_loss']]

    eval_log_dict['val_itm_loss'] = [log['val_itm_loss'] for log in eval_log]
    eval_log_dict['val_itc_loss'] = [log['val_itc_loss'] for log in eval_log]

    score_dict['r1'] = [log['r1'] for log in eval_log]
    score_dict['r5'] = [log['r5'] for log in eval_log]
    score_dict['r10'] = [log['r10'] for log in eval_log]
    score_dict['bleu_score'] = [log['bleu_score'] for log in eval_log]

    train_iter =[i for i in range(len(train_log_dict['itm_loss']))]
    val_epoch = [i for i in range(len(eval_log_dict['val_itm_loss']))]

    # train loss plot
    fig, axes = plt.subplots(2, 1, figsize=(7, 9))

    plot_data = [
        [train_log_dict['itm_loss'], train_log_dict['itc_loss']],
        [train_log_dict['caption_loss']]
    ]
    labels = [['itm_loss', 'itc_loss'], ['cap_loss']]
    colors = [['#377eb8', 'orange'], ['green']]
    titles = ["ITC and ITM Loss values", "Caption Loss values"]

    for i, ax in enumerate(axes):
        for data, label, color in zip(plot_data[i], labels[i], colors[i]):
            ax.plot(train_iter, data, label=label, color=color)
        ax.set(xlabel='iter', ylabel='loss', title=titles[i])
        ax.legend()
        ax.grid(True)
    plt.savefig(f'{save_dir}/train_loss.jpeg')

    # eval loss plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    title = ["EVAL ITM Loss values", "EVAL ITC Loss values"]
    for i, k in enumerate(eval_log_dict):
        axes[i].plot(val_epoch, eval_log_dict[k], label=labels[0][i], color=colors[0][i])
        axes[i].set(xlabel='epoch', ylabel='loss', title=title[i])
        axes[i].legend()
        axes[i].grid(True)
    plt.savefig(f'{save_dir}/eval_loss.jpeg')

    # score plot
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    titles = ['recall@1', 'recall@5', 'recall@10', 'bleu_score']
    for i, k in enumerate(score_dict):
        axes[i].plot(val_epoch, score_dict[k], label=titles[i], color='#377eb8')
        axes[i].set(xlabel='epoch', ylabel='score', title=titles[i])
        axes[i].legend()
        axes[i].grid(True)

    plt.savefig(f'{save_dir}/eval_score.jpeg')
    plt.tight_layout()
    plt.show()


class EarlyStopping:
    def __init__(self, monitor='loss', patience=7, verbose=False, delta=0):

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

        if monitor == 'loss':
            self.mode = 'min'
            self.monitor_op = lambda a, b: a < b - self.delta
            self.best_score = float('inf')
        elif monitor == 'score':
            self.mode = 'max'
            self.monitor_op = lambda a, b: a > b + self.delta
            self.best_score = -float('inf')

    def __call__(self, current_value):
        if self.best_score is None:
            self.best_score = current_value

        elif self.monitor_op(current_value, self.best_score):
            self.best_score = current_value

            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
