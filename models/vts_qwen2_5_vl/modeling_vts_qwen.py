import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.utils import ModelOutput

from qwen_vl_utils import process_vision_info
from dataclasses import dataclass
from typing import Optional, Tuple, List

from .configuration_vts_qwen import VTS_Qwen2_5_VLConfig
from models.module.vts_module import QueryGuidedTokenSelector

@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class VTS_Qwen2_5_VL(Qwen2_5_VLForConditionalGeneration):
    config_class = VTS_Qwen2_5_VLConfig
    def __init__(self, config):
        super().__init__(config)
        self.rope_deltas = None
        self.stage = 'LoRA'  # 'VTS_full', 'LoRA'
        self.hard_select = False
        self.use_vts = True
        self.frame_level = False
        self.parameters_free=False
        if self.use_vts:
            self.vts_module = QueryGuidedTokenSelector(D_vis=config.vision_config.out_hidden_size,  
                                                D_q=config.hidden_size, 
                                                D_r=config.VTS_hidden_size, 
                                                token_ratio=config.VTS_token_ratio, 
                                                temp=config.VTS_temp, 
                                                parameters_free=self.parameters_free,
                                                use_att_pool=False, 
                                                post_mlp_enabled=not self.parameters_free,
                                                image_token_id = config.image_token_id,
                                                video_token_id = config.video_token_id) 

        embed_dim = self.model.embed_tokens.embedding_dim
        vis_dim = getattr(config, "out_hidden_size", None) or getattr(config, "visual_hidden_size", None) or config.hidden_size
        if vis_dim != embed_dim:
            self.video_proj = nn.Linear(vis_dim, embed_dim, bias=False)
        else:
            self.video_proj = None

        if self.stage == 'VTS_full' and self.use_vts:
            self.freeze_all_parameters()
            self.unfreeze_vts_module()
        elif self.stage == 'LoRA':
            self.unfreeze_all_parameters()
        else:
            raise ValueError(f"Unknown training stage: {self.stage}")

    def freeze_all_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        print(f"Frozen: All")
    
    def unfreeze_all_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
        print(f"Unfrozen: All")
    
    def unfreeze_vts_module(self):
        for name, param in self.vts_module.named_parameters():
            param.requires_grad = True
            print(f"Unfrozen: {name}")

    def get_accepted_params(self, model_class):
        """Get the list of parameters accepted by the base class's forward method."""
        import inspect
        signature = inspect.signature(model_class.forward)
        return set(signature.parameters.keys())
  
    def filter_and_pad_inputs(self, global_keep_mask, position_ids, inputs_embeds, 
                              cache_position=None, pad_value=0.0):
        """
        Filter and pad input tensors according to global_keep_mask.
        Returns:
            filtered_inputs_embeds: (B, max_keep_len, D)
            filtered_position_ids:  (3, B, max_keep_len)
            new_attention_mask:     (B, max_keep_len)
            new_cache_position:     None or (max_keep_len,)
        """
        B, seq_len = global_keep_mask.shape
        D = inputs_embeds.size(-1)

        filtered_embeds_list = []
        filtered_pos_ids_list = []
        new_attention_masks = []

        for b in range(B):
            keep = global_keep_mask[b].bool()
            embeds_b = inputs_embeds[b][keep]
            pos_b = position_ids[:, b, keep]
            filtered_embeds_list.append(embeds_b)
            filtered_pos_ids_list.append(pos_b)
            new_attention_masks.append(torch.ones(len(embeds_b), dtype=torch.long, device=embeds_b.device))

        max_keep_len = max(len(x) for x in filtered_embeds_list)

        filtered_inputs_embeds = pad_sequence(filtered_embeds_list, batch_first=True, padding_value=pad_value)
        new_attention_mask = pad_sequence(new_attention_masks, batch_first=True, padding_value=0)
        filtered_position_ids = torch.zeros((3, B, max_keep_len), dtype=position_ids.dtype, device=position_ids.device)
        for b, pos in enumerate(filtered_pos_ids_list):
            filtered_position_ids[:, b, :pos.shape[1]] = pos

        if cache_position is not None:
            new_cache_position = torch.arange(max_keep_len, device=inputs_embeds.device)
        else:
            new_cache_position = None

        return filtered_inputs_embeds, filtered_position_ids, new_attention_mask, new_cache_position

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        timestamps=None,
        video_durations=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
    ):
        """
        Forward pass with integrated VTS module.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    print(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                    if n_video_features > n_video_tokens:
                        video_embeds = video_embeds[:n_video_tokens, :]
                        print("Truncate redundant features")
                    else:
                        pad_len = n_video_tokens - n_video_features
                        pad = torch.zeros(video_embeds.size(0), pad_len, video_embeds.size(2), device=video_embeds.device, dtype=video_embeds.dtype)
                        video_embeds = torch.cat([video_embeds, pad], dim=1)
                        print("Filling in missing features")

                if self.use_vts:
                    if self.frame_level:
                        n = torch.div(video_grid_thw[:, 1] * video_grid_thw[:, 2], 4, rounding_mode='floor').long()
                        embeds_grid_tn = torch.stack([video_grid_thw[:, 0].long(), n], dim=1)
                    else:
                        embeds_grid_tn = None
                    video_embeds_sel, video_topk_mask, _ = self.vts_module(video_embeds, inputs_embeds, input_ids, embeds_grid_tn=embeds_grid_tn)
                else:
                    video_embeds_sel = video_embeds
                    N_vis = video_embeds_sel.shape[0]
                    num_topk = int(N_vis * self.config.VTS_token_ratio)
                    indices = torch.randperm(N_vis)
                    video_topk_mask = torch.zeros(N_vis, dtype=torch.float32, device=video_embeds_sel.device)
                    video_topk_mask[indices[:num_topk]] = 1.0

                # Global token keep mask: non-video tokens default to kept (1.0)
                mask = input_ids == self.config.video_token_id
                mask = mask.to(inputs_embeds.device)
                video_topk_mask = video_topk_mask.to(inputs_embeds.device)
                
                global_keep_mask = torch.ones_like(input_ids, dtype=video_topk_mask.dtype, device=inputs_embeds.device)
                global_keep_mask = global_keep_mask.masked_scatter(mask, video_topk_mask)

                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)

                mask_expanded = mask_expanded.to(inputs_embeds.device)
                video_embeds_sel = video_embeds_sel.to(inputs_embeds.device, inputs_embeds.dtype)

                inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, video_embeds_sel) 
            else:
                global_keep_mask = torch.ones_like(input_ids, dtype=torch.long, device=inputs_embeds.device)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or getattr(past_key_values, "get_seq_length", lambda: 0)() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        position_ids = position_ids.to(inputs_embeds.device)
        if self.hard_select:
            filtered_embeds, filtered_pos, new_mask, new_cache_pos = self.filter_and_pad_inputs(
                global_keep_mask, position_ids, inputs_embeds, cache_position
            )

            outputs = self.model(
                input_ids=None,
                position_ids=filtered_pos,
                attention_mask=new_mask,
                past_key_values=past_key_values,
                inputs_embeds=filtered_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=new_cache_pos
            )

        else:
            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=global_keep_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position
            )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            if shift_logits.shape[1] != shift_labels.shape[1]:
                pad_len = shift_logits.shape[1] - shift_labels.shape[1]
                if pad_len > 0:
                    pad = torch.full((shift_labels.shape[0], pad_len), -100, dtype=shift_labels.dtype, device=shift_labels.device)
                    shift_labels = torch.cat([shift_labels, pad], dim=1)
                else:
                    shift_labels = shift_labels[:, :shift_logits.shape[1]]

            shift_labels = shift_labels.contiguous()

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


AutoModelForCausalLM.register(VTS_Qwen2_5_VLConfig, VTS_Qwen2_5_VL)
