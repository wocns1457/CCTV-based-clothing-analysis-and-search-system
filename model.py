from typing import Optional, Dict

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BlipConfig, BlipForConditionalGeneration, BlipForImageTextRetrieval


class CCTV_Analysis_Model(nn.Module):
    def __init__(self, load_pretrained_model=False):
        super().__init__()

        if load_pretrained_model:
            caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            retrieval_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        else:
            caption_config = BlipConfig.from_pretrained("Salesforce/blip-image-captioning-base")
            retrieval_config = BlipConfig.from_pretrained("Salesforce/blip-itm-base-coco")
            caption_model = BlipForConditionalGeneration(caption_config)
            retrieval_model = BlipForImageTextRetrieval(retrieval_config)

        self.vision_model = retrieval_model.vision_model

        self.text_encoder = retrieval_model.text_encoder
        self.text_decoder = caption_model.text_decoder

        self.vision_proj = retrieval_model.vision_proj
        self.text_proj = retrieval_model.text_proj
        self.itm_head = retrieval_model.itm_head

        self.decoder_input_ids = caption_model.decoder_input_ids  # 30522
        self.decoder_pad_token_id = caption_model.decoder_pad_token_id # 0
        self.decoder_eos_token_id = caption_model.config.text_config.eos_token_id  # 2
        self.decoder_sep_token_id = caption_model.config.text_config.sep_token_id # 102

    def forward(self,
                pixel_values: torch.FloatTensor,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                image_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                neg_input_ids: Optional[torch.LongTensor] = None,
                neg_attention_mask: Optional[torch.LongTensor] = None,
                return_loss: Optional[bool] = True,
                task: Optional[str] = None,
                label_smoothing: Optional[float] = 0.0,
                margin: Optional[float] = 0.0,
                text_similarity_threshold: Optional[float] = 1.0,
                ) -> Dict:

        itm_loss = None
        itc_loss = None
        caption_loss = None

        if image_embeds is None:
            vision_outputs = self.vision_model(pixel_values=pixel_values,
                                               output_attentions=None,
                                               output_hidden_states=None,
                                               interpolate_pos_encoding=False)
            image_embeds = vision_outputs[0]

        batch_size = image_embeds.shape[0]

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        output_dict = {'image_embeds':image_embeds}

        if task == 'itm' or task is None:
            itm_pos_embeds = self.text_encoder(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               encoder_hidden_states=image_embeds,
                                               encoder_attention_mask=image_atts)[0]

            if return_loss:
                if neg_input_ids is None and neg_attention_mask is None:
                    pos_embeddings = self.text_encoder(input_ids,
                                                       attention_mask=attention_mask)[0][:, 0, :]

                    # Cosine similarity를 이용하여 텍스트 유사도 행렬 생성
                    text_similarity = F.cosine_similarity(pos_embeddings.unsqueeze(0), pos_embeddings.unsqueeze(1))

                    # Permutation 생성 및 텍스트 유사도 기반 negative sample 조정
                    perm = torch.randperm(batch_size)
                    neg_input_ids = input_ids[perm]
                    neg_attention_mask = attention_mask[perm]

                    for i in range(batch_size):
                        # 유사도가 높을 경우 다른 샘플을 선택
                        while text_similarity[i, perm[i]] > text_similarity_threshold:  # 예: 유사도가 0.85 이상일 경우 변경
                            perm[i] = (perm[i] + 1) % batch_size
                        neg_input_ids[i] = input_ids[perm[i]]
                        neg_attention_mask[i] = attention_mask[perm[i]]

                itm_neg_embeds = self.text_encoder(input_ids=neg_input_ids,
                                                   attention_mask=neg_attention_mask,
                                                   encoder_hidden_states=image_embeds,
                                                   encoder_attention_mask=image_atts)[0]

                question_embeddings = torch.cat([itm_pos_embeds[:,0,:], itm_neg_embeds[:,0,:]], dim=0)

                itm_outputs = self.itm_head(question_embeddings)

                itm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long),
                                        torch.zeros(batch_size, dtype=torch.long)], dim=0).to(itm_outputs.device)
                itm_loss = F.cross_entropy(itm_outputs, itm_labels, label_smoothing=label_smoothing)
            else:
                question_embeddings = itm_pos_embeds[:,0,:]
                itm_outputs = self.itm_head(question_embeddings)

            output_dict['itm'] = {'loss': itm_loss, 'outputs': itm_outputs}


        if task == 'itc' or task is None:
            question_embeds = self.text_encoder(input_ids=input_ids,
                                                attention_mask=attention_mask)[0]

            image_embeds_norm = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_embeds_norm = F.normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

            similarity = text_embeds_norm @ image_embeds_norm.t()

            if return_loss:
                text_similarity = text_embeds_norm @ text_embeds_norm.T

                # 텍스트 유사도 임계값을 초과하는 인덱스 쌍을 찾음
                similar_text_pairs = (text_similarity > text_similarity_threshold).nonzero(as_tuple=True)
                similar_text_pairs = [(i.item(), j.item()) for i, j in zip(*similar_text_pairs) if i != j]

                exclude_mask = torch.zeros_like(similarity, dtype=torch.bool)

                # exclude_mask에서 유사한 쌍을 표시
                for i, j in similar_text_pairs:
                    exclude_mask[i, j] = True
                    exclude_mask[j, i] = True

                # 마스크를 적용하여 제외된 쌍에 낮은 값 설정
                masked_similarity = similarity.masked_fill(exclude_mask, -1e4)

                text_loss = F.cross_entropy(masked_similarity, torch.arange(batch_size, device=similarity.device))
                image_loss = F.cross_entropy(masked_similarity.T, torch.arange(batch_size, device=similarity.device))

                contrastive_loss1 = (text_loss + image_loss) / 2

                # contrastive loss Hard Negative Mining
                contrastive_loss2 = 0

                mask = text_similarity > text_similarity_threshold
                positive_sim = similarity.diag()

                for i in range(batch_size):
                    pos_sim = positive_sim[i]
                    hard_negatives = similarity[i][(~mask[i]) & (similarity[i] + margin > pos_sim)]
                    for hard_neg in hard_negatives:
                        contrastive_loss2 += F.relu(margin + hard_neg - pos_sim)

                contrastive_loss2 =  contrastive_loss2 / batch_size

                itc_loss = contrastive_loss1*0.5 + contrastive_loss2*0.5

            output_dict['itc'] = {'loss':itc_loss, 'image_embeds_norm':image_embeds_norm, 'text_embeds_norm':text_embeds_norm, 'similarity':similarity}


        if task == 'caption' or task is None:
            outputs = self.text_decoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        encoder_hidden_states=image_embeds,
                                        labels=labels,
                                        reduction="mean")

            caption_loss = outputs.loss

            output_dict['caption_loss'] = caption_loss

        return output_dict


    @torch.no_grad()
    def generate(self,
                pixel_values: Optional[torch.FloatTensor] = None,
                image_embeds: Optional[torch.FloatTensor] = None,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.LongTensor] = None,
                **generate_kwargs
                ) -> torch.LongTensor:

        if image_embeds is None:
            vision_outputs = self.vision_model(pixel_values=pixel_values,
                                               interpolate_pos_encoding=False)
            image_embeds = vision_outputs[0]

        batch_size = image_embeds.shape[0]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.decoder_input_ids, self.decoder_eos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        input_ids[:, 0] = self.decoder_input_ids
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        outputs = self.text_decoder.generate(input_ids=input_ids[:, :-1],
                                             eos_token_id=self.decoder_sep_token_id,
                                             pad_token_id=self.decoder_pad_token_id,
                                             attention_mask=attention_mask,
                                             encoder_hidden_states=image_embeds,
                                             encoder_attention_mask=image_attention_mask,
                                             **generate_kwargs,
                                             )

        return outputs