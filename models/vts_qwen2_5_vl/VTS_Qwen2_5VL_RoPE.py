import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


class QueryGuidedTokenSelector(nn.Module):
    """
    VTS Module
    """
    def __init__(self, D_vis, D_q, D_r=256, token_ratio=0.25, temp=1.0, use_att_pool=True, sel_mode='soft', agg_mode='mixed'):
        super().__init__()
        self.Wf = nn.Linear(D_vis, D_r, bias=False)
        self.Wq = nn.Linear(D_q, D_r, bias=False)
        self.token_ratio = token_ratio
        self.temp = temp
        self.sel_mode = sel_mode    # "hard" or "soft"
        self.agg_mode = agg_mode    # "sim", "uniform", "mixed", "index"
        self.use_att_pool = use_att_pool
        if use_att_pool:
            self.q_pool_att = nn.Sequential(
                nn.Linear(D_q, D_q),
                nn.Tanh(),
                nn.Linear(D_q, 1)
            )

        self.post_mlp = nn.Sequential(
            nn.Linear(D_vis, D_vis),
            nn.ReLU(),
            nn.Linear(D_vis, D_vis)
        )

    def pool_query(self, q):
        # q: [B, L, D_q]
        if not self.use_att_pool:
            return q.mean(dim=1, keepdim=True)
        a = self.q_pool_att(q)
        a = torch.softmax(a, dim=1)
        return (a * q).sum(dim=1, keepdim=True)

    def forward(self, f, q):
        # f: [B, THW, D_vis], q: [B, L, D_q]
        B, THW, D = f.shape
        q_pool = self.pool_query(q)
        f_p = self.Wf(f)
        q_p = self.Wq(q_pool).squeeze(1)
        s = torch.einsum('btd,bd->bt', f_p, q_p)
        soft_w = F.softmax(s / self.temp, dim=1)
        K = int(THW*self.token_ratio)
        topk = torch.topk(soft_w, k=K, dim=1)
        idx = topk.indices

        if self.sel_mode == 'hard':
            hard_mask = torch.zeros_like(soft_w).scatter(1, idx, 1.0)
            # Straight-through trick: gradients flow through soft_w
            weights = hard_mask + (soft_w - soft_w.detach())

            idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
            f_sel = torch.gather(f, dim=1, index=idx_exp)

            idx_exp = idx.unsqueeze(-1)
            w_sel = torch.gather(weights.unsqueeze(-1), dim=1, index=idx_exp)

            f_sel_processed = self.post_mlp(f_sel) * w_sel

        elif self.sel_mode == "soft":
            # Soft top-K: unselected tokens are aggregated into selected ones via soft weights
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
            f_sel = torch.gather(f, dim=1, index=idx_exp)

            not_sel_mask = torch.ones_like(soft_w)
            not_sel_mask.scatter_(1, idx, 0.0)
            not_sel_w = soft_w * not_sel_mask

            if self.agg_mode == "sim":  
                # Similarity-based assignment
                f_sel_proj = torch.gather(f_p, 1, idx.unsqueeze(-1).expand(-1, -1, f_p.size(-1)))
                sim = torch.einsum('btd,bkd->btk', f_p, f_sel_proj)
                assign = torch.softmax(sim, dim=-1)
                agg = torch.einsum('bt,btd,btk->bkd', not_sel_w, f, assign)
                f_sel_processed = self.post_mlp(f_sel) + agg

            elif self.agg_mode == "uniform":  
                # Uniform distribution across selected tokens
                agg_vec = torch.einsum('bt,btd->bd', not_sel_w, f)
                agg = agg_vec.unsqueeze(1).expand(-1, K, -1) / K
                f_sel_processed = self.post_mlp(f_sel) + agg

            elif self.agg_mode == "mixed":  
                # Weighted distribution using soft weights
                agg_vec = torch.einsum('bt,btd->bd', not_sel_w, f)
                w_sel = torch.gather(soft_w, 1, idx)
                agg = agg_vec.unsqueeze(1) * w_sel.unsqueeze(-1)
                f_sel_processed = self.post_mlp(f_sel) + agg

            elif self.agg_mode == "index":  
                # Nearest-neighbor assignment by position
                device = f.device
                all_pos = torch.arange(THW, device=device).unsqueeze(0).expand(B, -1)
                idx_sorted, _ = torch.sort(idx, dim=1)
                dist = (all_pos.unsqueeze(-1) - idx_sorted.unsqueeze(1)).abs()
                nearest = dist.argmin(dim=-1)
                assign = F.one_hot(nearest, num_classes=K).float()
                agg = torch.einsum('bt,btd,btk->bkd', not_sel_w, f, assign)
                f_sel_processed = self.post_mlp(f_sel) + agg

            else:
                raise ValueError(f"Unknown agg_mode: {self.agg_mode}")

            w_sel = torch.gather(soft_w, 1, idx)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return f_sel_processed, idx, w_sel.squeeze(-1), soft_w


class VTS_Qwen2_5_VL(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        self.vts_module = QueryGuidedTokenSelector(D_vis=config.vision_config.out_hidden_size,  
                                                D_q=config.hidden_size, 
                                                D_r=config.VTS_hidden_size, 
                                                token_ratio=config.VTS_token_ratio, 
                                                temp=config.VTS_temp, 
                                                use_att_pool=config.VTS_use_att_pool, 
                                                sel_mode=config.VTS_sel_mode, 
                                                agg_mode=config.VTS_agg_mode)
        
        embed_dim = self.model.embed_tokens.embedding_dim
        vis_dim = getattr(config, "out_hidden_size", None) or getattr(config, "visual_hidden_size", None) or config.hidden_size
        if vis_dim != embed_dim:
            self.video_proj = nn.Linear(vis_dim, embed_dim, bias=False)
        else:
            self.video_proj = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        cache_position=None,
        **kwargs
    ):
        """
        Forward with RoPE-aware VTS token selection:
        - Compute full position_ids via get_rope_index during prefill
        - VTS selects K video tokens and returns relative indices
        - Map indices to global positions and construct new inputs_embeds + position_ids
        - Pad batch to max_len and call self.model
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

            position_ids_full = None
            rope_deltas = None
            prefill_condition = (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or getattr(past_key_values, "get_seq_length", lambda: 0)() == 0)
            )
            if prefill_condition:
                position_ids_full, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas

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

                video_token_mask = (input_ids == self.config.video_token_id)
                B, seq_len = input_ids.shape
                lens = video_token_mask.sum(dim=1).tolist()

                if sum(lens) != video_embeds.shape[0]:
                    raise ValueError("video_embeds length mismatch with token mask counts")
                video_splits = list(torch.split(video_embeds, lens, dim=0)) if len(lens) > 0 else [torch.zeros((0, video_embeds.shape[-1]), device=video_embeds.device)]

                max_len = max(lens) if len(lens) > 0 else 0
                D_vis = video_embeds.shape[-1] if video_embeds.numel() else 0
                device = video_embeds.device
                if max_len == 0:
                    pass
                else:
                    padded = torch.zeros((B, max_len, D_vis), device=device, dtype=video_embeds.dtype)
                    valid_lens = []
                    for i, ve in enumerate(video_splits):
                        li = ve.shape[0]
                        valid_lens.append(li)
                        if li > 0:
                            padded[i, :li] = ve

                    main_dtype = self.model.embed_tokens.weight.dtype
                    padded = padded.to(device=self.model.device, dtype=main_dtype)
                    inputs_embeds = inputs_embeds.to(device=self.model.device, dtype=main_dtype)
                    f_sel, idx, w_sel, soft_w = self.vts_module(padded, inputs_embeds)

                    new_inputs_embeds_list = []
                    new_position_ids_list = []
                    new_attention_mask_list = []
                    pos_full = position_ids_full
                    if pos_full is not None and pos_full.ndim == 2:
                        pos_full = pos_full.unsqueeze(0).expand(3, -1, -1)

                    for b in range(B):
                        Li = valid_lens[b]
                        video_pos = torch.nonzero(video_token_mask[b], as_tuple=False).squeeze(1)
                        if video_pos.numel() == 0:
                            new_emb = inputs_embeds[b]
                            if pos_full is not None:
                                pos_b = pos_full[:, b, :]
                            else:
                                pos_b = None
                            new_inputs_embeds_list.append(new_emb)
                            new_position_ids_list.append(pos_b)
                            new_attention_mask_list.append(torch.ones(new_emb.shape[0], device=device, dtype=torch.long))
                            continue

                        start = int(video_pos[0].item())
                        end = int(video_pos[-1].item())
                        contiguous = (video_pos.numel() == (end - start + 1))
                        idx_b = idx[b]
                        valid_mask = (idx_b < Li)
                        idx_b_valid = idx_b[valid_mask]
                        if idx_b_valid.numel() > 0:
                            sort_order = torch.argsort(idx_b_valid)
                            idx_b_valid = idx_b_valid[sort_order]
                            f_sel_b = f_sel[b][valid_mask][sort_order]
                        else:
                            idx_b_valid = torch.tensor([], dtype=torch.long, device=device)
                            f_sel_b = torch.zeros((0, D_vis), device=device, dtype=f_sel.dtype)

                        if idx_b_valid.numel() > 0:
                            video_pos = video_pos.to(idx_b_valid.device)
                            global_video_positions = video_pos[idx_b_valid]
                            global_video_positions = global_video_positions.to(pos_full.device) if pos_full is not None and global_video_positions.numel() > 0 else global_video_positions
                        else:
                            global_video_positions = torch.tensor([], dtype=torch.long, device=device)

                        if contiguous:
                            before_emb = inputs_embeds[b, :start, :]
                            after_emb = inputs_embeds[b, end+1:, :]
                            if self.video_proj is not None and f_sel_b.numel() > 0:
                                f_sel_b_proj = self.video_proj(f_sel_b)
                            else:
                                f_sel_b_proj = f_sel_b if f_sel_b.numel() > 0 else torch.zeros((0, self.model.embed_tokens.embedding_dim), device=device)
                            target_device = inputs_embeds.device
                            before_emb = before_emb.to(target_device)
                            f_sel_b_proj = f_sel_b_proj.to(target_device)
                            after_emb = after_emb.to(target_device)
                            new_emb = torch.cat([before_emb, f_sel_b_proj, after_emb], dim=0)
                            new_mask = torch.cat([torch.ones(before_emb.shape[0], device=device, dtype=torch.long),
                                                  torch.ones(f_sel_b_proj.shape[0], device=device, dtype=torch.long),
                                                  torch.ones(after_emb.shape[0], device=device, dtype=torch.long)], dim=0)
                            if pos_full is not None:
                                pos_before = pos_full[:, b, :start]
                                pos_selected = pos_full[:, b, global_video_positions] if global_video_positions.numel()>0 else pos_full[:, b, 0:0]
                                pos_after = pos_full[:, b, end+1:]
                                new_pos_b = torch.cat([pos_before, pos_selected, pos_after], dim=1)
                            else:
                                new_pos_b = None
                        else:
                            # Non-contiguous video tokens: iterate and reconstruct
                            new_emb_chunks = []
                            new_mask_chunks = []
                            new_pos_chunks = []
                            video_inserted = False
                            for i in range(seq_len):
                                tokid = input_ids[b, i].item()
                                if tokid == self.config.video_token_id:
                                    if not video_inserted:
                                        if self.video_proj is not None and f_sel_b.numel() > 0:
                                            f_sel_b_proj = self.video_proj(f_sel_b)
                                        else:
                                            f_sel_b_proj = f_sel_b if f_sel_b.numel() > 0 else torch.zeros((0, self.model.embed_tokens.embedding_dim), device=device)
                                        new_emb_chunks.append(f_sel_b_proj)
                                        new_mask_chunks.append(torch.ones(f_sel_b_proj.shape[0], device=device, dtype=torch.long))
                                        if pos_full is not None:
                                            pos_selected = pos_full[:, b, global_video_positions] if global_video_positions.numel()>0 else pos_full[:, b, 0:0]
                                            new_pos_chunks.append(pos_selected)
                                        video_inserted = True
                                    else:
                                        continue
                                else:
                                    new_emb_chunks.append(inputs_embeds[b, i:i+1, :])
                                    new_mask_chunks.append(torch.ones(1, device=device, dtype=torch.long))
                                    if pos_full is not None:
                                        new_pos_chunks.append(pos_full[:, b, i:i+1])
                            if len(new_emb_chunks) > 0:
                                new_emb = torch.cat(new_emb_chunks, dim=0)
                                new_mask = torch.cat(new_mask_chunks, dim=0)
                            else:
                                new_emb = torch.zeros((0, self.model.embed_tokens.embedding_dim), device=device)
                                new_mask = torch.zeros((0,), device=device, dtype=torch.long)
                            new_pos_b = torch.cat(new_pos_chunks, dim=1) if pos_full is not None and len(new_pos_chunks) > 0 else None

                        new_inputs_embeds_list.append(new_emb)
                        new_position_ids_list.append(new_pos_b)
                        new_attention_mask_list.append(new_mask)

                    new_lens = [x.shape[0] for x in new_inputs_embeds_list]
                    max_new_len = max(new_lens)
                    E = self.model.embed_tokens.embedding_dim
                    new_inputs_embeds = torch.zeros((B, max_new_len, E), device=device, dtype=inputs_embeds.dtype)
                    new_attention_mask = torch.zeros((B, max_new_len), device=device, dtype=torch.long)
                    if position_ids_full is not None:
                        pos_ids_padded = torch.zeros((3, B, max_new_len), device=device, dtype=position_ids_full.dtype)
                    else:
                        pos_ids_padded = None

                    for b in range(B):
                        l = new_inputs_embeds_list[b].shape[0]
                        if l > 0:
                            new_inputs_embeds[b, :l, :] = new_inputs_embeds_list[b]
                            new_attention_mask[b, :l] = new_attention_mask_list[b]
                            if pos_ids_padded is not None and new_position_ids_list[b] is not None:
                                pos_ids_padded[:, b, :l] = new_position_ids_list[b]
                        else:
                            continue

                    inputs_embeds = new_inputs_embeds
                    attention_mask = new_attention_mask
                    position_ids = pos_ids_padded

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

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Pad shift_labels to match shift_logits length
            if shift_logits.shape[1] != shift_labels.shape[1]:
                pad_len = shift_logits.shape[1] - shift_labels.shape[1]
                if pad_len > 0:
                    pad = torch.full((shift_labels.shape[0], pad_len), -100, dtype=shift_labels.dtype, device=shift_labels.device)
                    shift_labels = torch.cat([shift_labels, pad], dim=1)
                else:
                    shift_labels = shift_labels[:, :shift_logits.shape[1]]

            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
