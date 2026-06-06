import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, List, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPast
from dataclasses import dataclass

from transformers import InternVLModel, InternVLForConditionalGeneration
from .internvl_3.modeling_internvl_chat import InternVLChatModel
from .internvl_3.configuration_internvl_chat import InternVLChatConfig
from .configuration_vts_intern import VTS_InternVL_3Config
from models.module.vts_module import QueryGuidedTokenSelector

@dataclass
class InternVLModelOutputWithPast(BaseModelOutputWithPast):
    image_hidden_states: Optional[torch.FloatTensor] = None

@dataclass
class InternVLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None

class VTS_InternVL_3Core(InternVLModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.frame_level = False

        self.vts_module = QueryGuidedTokenSelector(D_vis=config.vision_config.hidden_size,  
                                                D_q=config.text_config.hidden_size, 
                                                D_r=config.VTS_hidden_size, 
                                                token_ratio=config.VTS_token_ratio, 
                                                temp=config.VTS_temp, 
                                                parameters_free=False,
                                                use_att_pool=False, 
                                                post_mlp_enabled=True,
                                                image_token_id = config.image_token_id,
                                                video_token_id = config.image_token_id) 
    
    def apply_qts_selection(self, vit_embeds, inputs_embeds, input_ids):
        """Apply VTS selection strategy."""
        B, N, C = inputs_embeds.shape

        selected = (input_ids == self.config.image_token_id).reshape(-1)
        if self.frame_level:
            frame_num = vit_embeds.shape[0] // B
            embeds_grid_tn = torch.tensor([[frame_num, self.config.image_seq_length]] * B, dtype=torch.long)
        else:
            embeds_grid_tn = None

        vit_embeds = vit_embeds.reshape(-1, vit_embeds.size(-1))
        
        video_embeds_sel, video_topk_mask, _ = self.vts_module(vit_embeds, inputs_embeds, input_ids, embeds_grid_tn=embeds_grid_tn)

        inputs_embeds = inputs_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)
        
        global_keep_mask = torch.ones_like(input_ids, dtype=video_topk_mask.dtype, device=inputs_embeds.device)
        selected = selected.to(inputs_embeds.device)
        video_topk_mask = video_topk_mask.to(inputs_embeds.device)
        global_keep_mask = global_keep_mask.masked_scatter(selected, video_topk_mask)
        global_keep_mask = global_keep_mask.reshape(B, N)

        video_embeds_sel = video_embeds_sel.to(inputs_embeds.device)
        video_pos = selected.unsqueeze(-1).expand(-1, inputs_embeds.size(1))
        video_embeds_sel = video_embeds_sel.to(inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_pos, video_embeds_sel.reshape(-1, C))

        inputs_embeds = inputs_embeds.reshape(B, N, C)
        
        return inputs_embeds, global_keep_mask

    def forward_core(self, 
                    pixel_values=None,
                    input_ids=None,
                    inputs_embeds=None,
                    attention_mask=None,
                    position_ids=None,
                    vision_feature_layer=None,
                    vision_feature_select_strategy=None,
                    past_key_values=None,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None):
        """
        VTS core processing flow.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            vit_embeds = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            vit_embeds = vit_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds, global_keep_mask = self.apply_qts_selection(vit_embeds, inputs_embeds, input_ids)
        else:
            global_keep_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=global_keep_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return InternVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=vit_embeds if pixel_values is not None else None,
        )

class VTS_InternVL_3(InternVLForConditionalGeneration):
    config_class = VTS_InternVL_3Config
    def __init__(self, config):
        super().__init__(config)
        self.stage = 'LoRA'  # 'VTS_full', 'LoRA'
        self.model = VTS_InternVL_3Core(config)

        if self.stage == 'VTS_full':
            self.freeze_all_parameters()
            self.unfreeze_vts_module()
        elif self.stage == 'LoRA':
            self.freeze_all_parameters()
            self.unfreeze_llm()
            self.unfreeze_vts_module()
            self.unfreeze_projector()
        else:
            raise ValueError(f"Unknown training stage: {self.stage}")

    def freeze_all_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        print("Frozen: All")
    
    def unfreeze_all_parameters(self):
        for name, param in self.named_parameters():
            param.requires_grad = True
        print("Unfrozen: All")
    
    def unfreeze_vts_module(self):
        for name, param in self.model.vts_module.named_parameters():
            param.requires_grad = True
            print(f"Unfrozen: {name}")
    
    def unfreeze_llm(self):
        for name, param in self.model.language_model.named_parameters():
            param.requires_grad = True
        print("Unfrozen: LLM")
    
    def unfreeze_projector(self):
        for name, param in self.model.multi_modal_projector.named_parameters():
            param.requires_grad = True
            print(f"Unfrozen: {name}")

    def freeze_vision(self):
        for name, param in self.model.vision_tower.named_parameters():
            param.requires_grad = False
            print(f"Frozen: {name}")

    def get_accepted_params(self, model_class):
        """Get the list of parameters accepted by the base class's forward method."""
        import inspect
        signature = inspect.signature(model_class.forward)
        return set(signature.parameters.keys())

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        past_key_values=None,
        labels=None,
        timestamps=None,
        video_durations=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep: Union[int, torch.Tensor] = 0
    ):
        """
        Forward pass with integrated VTS module.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        outputs = self.model.forward_core(
            pixel_values=pixel_values,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )

        return InternVLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        **kwargs,
    ):
        """
        Ensure pixel_values and other visual inputs are preserved during generation.
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


AutoModelForCausalLM.register(VTS_InternVL_3Config, VTS_InternVL_3)
