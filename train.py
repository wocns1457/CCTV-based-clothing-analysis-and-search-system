import os
import json
import yaml
import random
import argparse
from tqdm.auto import tqdm

import numpy as np

import evaluate

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets.utils import download_url

from torchvision import transforms

from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from cctv_utils import EarlyStopping, save_loss_plot
from data.make_dataset import create_peta_dataset, create_ai_hub_dataset, create_deepfashion_dataset
from model import CCTV_Analysis_Model


def train(model, optimizer, scheduler, device, main_train_loader, neg_text, processor, cfg, scaler=None):
    # train
    model.train()

    itm_loss = []
    itc_loss = []
    caption_loss = []

    for iteration, inputs in tqdm(enumerate(main_train_loader), total=len(main_train_loader), desc='Training'):
        inputs = inputs.to(device)
        bs = inputs['pixel_values'].size(0)

        if random.random() > 0.5:
            neg_sample = neg_text[iteration*bs: min(len(neg_text), (iteration+1)*bs)]
            neg_sample = processor(text=neg_sample, padding='max_length', max_length=40, truncation=True, return_tensors="pt").to(device)
        else:
            neg_sample = {'input_ids': None, 'attention_mask': None}

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=inputs['pixel_values'],
                                input_ids=inputs['input_ids'],
                                attention_mask=inputs['attention_mask'],
                                labels=inputs['input_ids'],
                                neg_input_ids=neg_sample['input_ids'],
                                neg_attention_mask=neg_sample['attention_mask'],
                                label_smoothing=cfg['label_smoothing'],
                                text_similarity_threshold=cfg['text_similarity_threshold'],
                                margin=cfg['margin'],
                                return_loss=True)

                total_loss = ((outputs['itm']['loss'] + outputs['itc']['loss']) / 2) * cfg['alpha'] + outputs['caption_loss'] * (1-cfg['alpha'])

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

        else:
            outputs = model(pixel_values=inputs['pixel_values'],
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=inputs['input_ids'],
                            neg_input_ids=neg_sample['input_ids'],
                            neg_attention_mask=neg_sample['attention_mask'],
                            label_smoothing=cfg['label_smoothing'],
                            text_similarity_threshold=cfg['text_similarity_threshold'],
                            margin=cfg['margin'],
                            return_loss=True)

            total_loss = ((outputs['itm']['loss'] + outputs['itc']['loss']) / 2) * cfg['alpha'] + outputs['caption_loss'] * (1-cfg['alpha'])

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

        itm_loss.append(outputs['itm']['loss'].item())
        itc_loss.append(outputs['itc']['loss'].item())
        caption_loss.append(outputs['caption_loss'].item())

    return itm_loss, itc_loss, caption_loss


@torch.no_grad()
def evaluation(model, data_loader, device, metric, cfg):
    # test
    model.eval()

    num_img = len(data_loader.dataset.datasets[0].data + data_loader.dataset.datasets[1].data)
    texts = data_loader.dataset.datasets[0].text + data_loader.dataset.datasets[1].text
    num_text = len(texts)

    processor = data_loader.dataset.datasets[0].processor

    text_ids, text_atts, text_embeds_norm = [], [], []
    image_embeds, image_embeds_norm = [], []
    itc_loss = []

    for inputs in data_loader:
        inputs.to(device)

        outputs = model(pixel_values=inputs['pixel_values'],
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        margin=cfg['margin'],
                        text_similarity_threshold=cfg['text_similarity_threshold'],
                        task='itc',
                        return_loss=True)

        itc_loss.append(outputs['itc']['loss'].item())
        image_embeds.append(outputs['image_embeds'])
        image_embeds_norm.append(outputs['itc']['image_embeds_norm'])

        text_embeds_norm.append(outputs['itc']['text_embeds_norm'])
        text_ids.append(inputs.input_ids)
        text_atts.append(inputs.attention_mask)

    image_embeds = torch.cat(image_embeds,dim=0)
    image_embeds_norm = torch.cat(image_embeds_norm,dim=0)

    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_embeds_norm = torch.cat(text_embeds_norm,dim=0)

    sims_matrix = text_embeds_norm @ image_embeds_norm.t()

    score_matrix_t2i = torch.full((num_text, num_img), -100.0).to(device)

    test_k = cfg['test_k']
    for i, sims in enumerate(sims_matrix):
        topk_sim, topk_idx = sims.topk(k=test_k, dim=0)
        encoder_output = image_embeds[topk_idx]

        output = model(pixel_values=None,
                       input_ids=text_ids[i].repeat(test_k, 1),
                       attention_mask=text_atts[i].repeat(test_k, 1),
                       image_embeds=encoder_output,
                       task='itm',
                       return_loss=False
                       )

        score = output['itm']['outputs'][:, 1]
        score_matrix_t2i[i, topk_idx] = score + topk_sim

    # 텍스트 유사도를 고려하여 Recall@k 계산
    recall_score = {1: 0, 5: 0, 10: 0}
    for i, score in enumerate(score_matrix_t2i):
        top_k_sim, top_k_inds = score.topk(k=10, dim=-1)
        reference_embed = text_embeds_norm[i]
        comparison_embeds = text_embeds_norm[top_k_inds]
        text_sim = F.cosine_similarity(reference_embed, comparison_embeds)

        if i in top_k_inds[:1] or (text_sim[:1] >= 0.9).sum() > 0:
            recall_score[1] += 1
        elif (text_sim[:1] >= 0.8).sum() > 0:
            recall_score[1] += 0.5

        if i in top_k_inds[:5] or (text_sim[:5] >= 0.9).sum() > 0:
            recall_score[5] += 1
        elif (text_sim[:5] >= 0.8).sum() > 0:
            recall_score[5] += 0.5

        if i in top_k_inds[:10] or (text_sim[:10] >= 0.9).sum() > 0:
            recall_score[10] += 1
        elif (text_sim[:10] >= 0.8).sum() > 0:
            recall_score[10] += 0.5

    total = len(score_matrix_t2i)
    recall_score = {k: v / total * 100 for k, v in recall_score.items()}

    itm_loss = []
    captions = []
    texts = [[text] for text in texts]
    text_bs = cfg['text_bs']

    for i in range(0, num_text, text_bs):
        text_id = text_ids[i: min(num_text, i+text_bs)]
        text_att = text_atts[i: min(num_text, i+text_bs)]
        image_embed = image_embeds[i: min(num_text, i+text_bs)]

        output = model(pixel_values=None,
                       input_ids=text_id,
                       attention_mask=text_att,
                       image_embeds = image_embed,
                       task='itm',
                       return_loss=False
                       )

        output = output['itm']['outputs']

        label = torch.ones(output.shape[0], dtype=torch.long, device=output.device)
        loss = F.cross_entropy(output, label)
        itm_loss.append(loss.item())

        caption = model.generate(image_embeds=image_embed, max_new_tokens=40)
        caption = processor.batch_decode(caption, skip_special_tokens=True)
        captions.extend(caption)

    bleu_score = metric.compute(predictions=captions, references=texts)['bleu'] * 100.0

    return itm_loss, itc_loss, recall_score, bleu_score
    

def main(cfg):
    SEED = cfg['SEED']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cudnn.benchmark = True

    prompt_config = json.load(open(cfg['prompt_config_path'], "r"))

    save_dir = cfg['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)], p=0.4),
                                    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.4),
                                    transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
                                    transforms.RandomVerticalFlip(p=0.3),
                                    transforms.RandomHorizontalFlip(p=0.3)
                                    ])

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    # PETA Dataset
    train_peta_dataset, val_peta_dataset, test_peta_dataset = create_peta_dataset(cfg['peta_dir'], transform, processor,
                                                                                  use_data_size=cfg['use_data_size'],
                                                                                  train_size=cfg['train_size'],
                                                                                  config=prompt_config,
                                                                                  seed=SEED)

    # AI HUB CCTV Dataset
    train_ai_hub_dataset, val_ai_hub_dataset, test_ai_hub_dataset= create_ai_hub_dataset(cfg['ai_hub_dir'], transform, processor,
                                                                                         use_data_size=cfg['use_data_size'],
                                                                                         train_size=cfg['train_size'],
                                                                                         config=prompt_config,
                                                                                         seed=SEED)

    # DeepFashsion Dataset
    train_deepfashion_dataset, _, _ = create_deepfashion_dataset(cfg['deepfashion_dir'], transform, processor, 
                                                                 use_data_size=int(cfg['use_data_size']/2),
                                                                train_size=cfg['train_size'],
                                                                seed=SEED)

    # ConcatDataset
    train_ConcatDataset = ConcatDataset([train_peta_dataset, train_ai_hub_dataset, train_deepfashion_dataset])
    val_ConcatDataset = ConcatDataset([val_peta_dataset, val_ai_hub_dataset])
    test_ConcatDataset = ConcatDataset([test_peta_dataset, test_ai_hub_dataset])

    train_Concat_loader = DataLoader(train_ConcatDataset, shuffle=True, pin_memory=True,
                                     batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    val_Concat_loader = DataLoader(val_ConcatDataset, shuffle=False, pin_memory=True,
                                   batch_size=cfg['batch_size']*2, num_workers=cfg['num_workers'])

    test_Concat_loader = DataLoader(test_ConcatDataset, shuffle=False, pin_memory=True,
                                    batch_size=cfg['batch_size']*2, num_workers=cfg['num_workers'])

    num_train_data = len(train_peta_dataset) + len(train_ai_hub_dataset) + len(train_deepfashion_dataset)

    coco_karpathy_train_url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
    download_url(coco_karpathy_train_url,'./')

    neg_sample_text = json.load(open('./coco_karpathy_train.json', 'r'))
    neg_sample_text = random.sample(neg_sample_text, num_train_data)
    neg_sample_text = [sample['caption'] for sample in neg_sample_text]

    # Model
    model = CCTV_Analysis_Model(load_pretrained_model=True)
    model.to(device)

    scaler = torch.amp.GradScaler() if cfg['mixed_precision'] else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])

    train_epoch = cfg['train_epoch']

    scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(cfg['warmup_epoch'] * train_epoch * len(train_Concat_loader)),
                    num_training_steps=len(train_Concat_loader) * train_epoch
                )

    early_stopping = EarlyStopping(monitor='score', patience=5, verbose=True)
    metric = evaluate.load("bleu")

    best_score = 0
    best_epoch = 0
    best_model = None

    for epoch in range(0, train_epoch):
        itm_loss, itc_loss, caption_loss = train(model, optimizer, scheduler, device, train_Concat_loader, neg_sample_text, processor, cfg, scaler=scaler)

        val_itm_loss, val_itc_loss, recall_score, bleu_score = evaluation(model, val_Concat_loader, device, metric, cfg)

        r_mean = (recall_score[1] + recall_score[5] + recall_score[10]) / 3

        print(f'TRAIN Epoch [{epoch}/{train_epoch-1}] itm loss : [{np.mean(itm_loss):.4f}], itc loss : [{np.mean(itc_loss):.4f}], caption loss : [{np.mean(caption_loss):.4f}]')
        print(f'EVAL Epoch [{epoch}/{train_epoch-1}] itm loss : [{np.mean(val_itm_loss):.4f}], itc loss : [{np.mean(val_itc_loss):.4f}], r@1 : [{recall_score[1]:.4f}], r@5 : [{recall_score[5]:.4f}], r@10 : [{recall_score[10]:.4f}], r_mean : [{r_mean:.4f}], bleu score : [{bleu_score:.4f}]')

        if r_mean > best_score:
            best_score = r_mean
            best_epoch = epoch
            best_model = model

        save_obj = {'model': best_model.state_dict(),
                    'epoch': best_epoch
                    }

        train_log = {'epoch': epoch,
                    'best_epoch': best_epoch,
                    'itm_loss': itm_loss,
                    'itc_loss': itc_loss,
                    'caption_loss': caption_loss,
                    }

        eval_log = {'epoch': epoch,
                    'best_epoch': best_epoch,
                    'val_itm_loss': np.mean(val_itm_loss),
                    'val_itc_loss': np.mean(val_itc_loss),
                    'r1' : recall_score[1],
                    'r5' : recall_score[5],
                    'r10' : recall_score[10],
                    'bleu_score' : bleu_score,
                    }

        with open(os.path.join(save_dir, "train_log.json"), "a") as f:
            f.write(json.dumps(train_log) + "\n")

        with open(os.path.join(save_dir, "eval_log.json"), "a") as f:
            f.write(json.dumps(eval_log) + "\n")


        early_stopping(r_mean)

        torch.cuda.empty_cache()

        train_Concat_loader.dataset.datasets[0].regenerate_prompt()
        train_Concat_loader.dataset.datasets[1].regenerate_prompt()

        if early_stopping.early_stop:
            print("Early stopping")
            break

    test_itm_loss, test_itc_loss, recall_score, bleu_score = evaluation(best_model, test_Concat_loader, device, metric, cfg)
    test_r_mean = (recall_score[1] + recall_score[5] + recall_score[10]) / 3

    print(f'TEST Best Epoch [{best_epoch}] itm loss : [{np.mean(test_itm_loss):.4f}], itc loss : [{np.mean(test_itc_loss):.4f}], r@1 : [{recall_score[1]:.4f}], r@5 : [{recall_score[5]:.4f}], r@10 : [{recall_score[10]:.4f}], r_mean : [{test_r_mean:.4f}], bleu score : [{bleu_score:.4f}]')

    if cfg['save_plot']:
        save_loss_plot(save_dir)

    torch.save(save_obj, os.path.join(save_dir, f'checkpoint_best_epoch{best_epoch}.pth'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_config', type=str, default='./train_config.yaml')
    args = parser.parse_args()

    config_path = args.train_config

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        print(config)

    main(config)
