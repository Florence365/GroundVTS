import torch
import torch.nn as nn
import torch.nn.functional as F

class QueryGuidedTokenSelector(nn.Module):
    """
    VTS Module
    """
    def __init__(self, 
                 D_vis, 
                 D_q, 
                 D_r, 
                 token_ratio=0.25, 
                 temp=1.0, 
                 parameters_free=False,
                 use_att_pool=False, 
                 post_mlp_enabled=True,
                 image_token_id: int = None,
                 video_token_id: int = None):
        super().__init__()
        self.D_vis = D_vis
        self.D_q = D_q
        self.D_r = D_r
        self.token_ratio = token_ratio
        self.temp = temp
        self.parameters_free = parameters_free
        self.use_att_pool = use_att_pool
        self.post_mlp_enabled = post_mlp_enabled
        self.batch_topk = True

        if not self.parameters_free:
            self.Wf = nn.Linear(max(D_vis, D_q), D_r, bias=False)
            self.Wq = nn.Linear(D_q, D_r, bias=False)
        
        if use_att_pool:
            self.q_pool_att = nn.Sequential(
                nn.Linear(D_q, D_q),
                nn.Tanh(),
                nn.Linear(D_q, 1)
            )

        if post_mlp_enabled:
            self.post_mlp = nn.Sequential(
                nn.Linear(max(D_vis, D_q), max(D_vis, D_q)),
                nn.ReLU(),
                nn.Linear(max(D_vis, D_q), max(D_vis, D_q)) 
            )
        else:
            self.post_mlp = None

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id


    # --- PREPROCESS helpers -------------------------------------------------
    def _split_video_embeds_by_input_ids(self, video_embeds, input_ids):
        """
        Split `video_embeds` (N_vis, D) into a list of per-sample tensors based on
        counts of video tokens in each row of input_ids.
        Returns:
            f_list: list of T_i x D tensors (len B)
            counts: list of T_i lengths
        """
        B, T_total = input_ids.shape
        assert self.video_token_id is not None, "video_token_id must be set"
        counts = (input_ids == self.video_token_id).long().sum(dim=1).tolist()

        f_list = []
        ptr = 0
        N_vis = video_embeds.shape[0]
        total_needed = sum(counts)

        # Truncate excess visual tokens
        if N_vis > total_needed:
            video_embeds = video_embeds[:total_needed]
            N_vis = total_needed
            
        for c in counts:
            if c == 0:
                f_list.append(torch.zeros((0, self.D_vis), dtype=video_embeds.dtype, device=video_embeds.device))
                continue
            if ptr + c > N_vis:
                remaining = max(N_vis - ptr, 0)
                part = video_embeds[ptr:ptr + remaining]
                pad = torch.zeros((c - remaining, self.D_vis), dtype=video_embeds.dtype, device=video_embeds.device)
                f_list.append(torch.cat([part, pad], dim=0))
                ptr = N_vis
            else:
                f_list.append(video_embeds[ptr:ptr + c])
                ptr += c
                
        return f_list, counts

    def _pad_list_to_tensor(self, list_of_tensors, pad_value=0.0):
        """
        Given list of [T_i, D] tensors, pad to [B, T_max, D] and return mask [B, T_max]
        If T_i == 0, row filled with pad_value and mask zeros.
        """
        B = len(list_of_tensors)
        D = list_of_tensors[0].shape[-1]
        lengths = [t.shape[0] for t in list_of_tensors]
        T_max = max(lengths) if len(lengths) > 0 else 0
        if T_max == 0:
            f_padded = torch.zeros((B, 0, D), device=list_of_tensors[0].device, dtype=list_of_tensors[0].dtype)
            mask = torch.zeros((B, 0), device=list_of_tensors[0].device, dtype=torch.long)
            return f_padded, mask, lengths
        out = []
        mask = []
        for t in list_of_tensors:
            T_i = t.shape[0]
            if T_i == 0:
                pad = t.new_full((T_max, D), pad_value)
                out.append(pad.unsqueeze(0))
                mask.append(t.new_zeros((T_max,), dtype=torch.long).unsqueeze(0))
            else:
                if T_i < T_max:
                    pad = t.new_full((T_max - T_i, D), pad_value)
                    row = torch.cat([t, pad], dim=0).unsqueeze(0)
                    m = torch.cat([t.new_ones((T_i,), dtype=torch.long), t.new_zeros((T_max - T_i,), dtype=torch.long)], dim=0).unsqueeze(0)
                else:
                    row = t.unsqueeze(0)
                    m = t.new_ones((T_i,), dtype=torch.long).unsqueeze(0)
                out.append(row)
                mask.append(m)
        f_padded = torch.cat(out, dim=0)  # [B, T_max, D]
        mask = torch.cat(mask, dim=0)     # [B, T_max]
        return f_padded, mask, lengths

    def _extract_queries_from_inputs(self, inputs_embeds, input_ids):
        """
        Extract query tokens (non-image, non-video tokens) from inputs_embeds.
        Returns q_list (list of [L_i, D_q]) and masks.
        We assume inputs_embeds' last dim == D_vis == D_q (same hidden size).
        """
        B, T_total, D = inputs_embeds.shape
        if self.image_token_id is None or self.video_token_id is None:
            raise RuntimeError("image_token_id and video_token_id must be set to extract queries")

        q_masks_bool = (input_ids != self.image_token_id) & (input_ids != self.video_token_id)
        q_list = []
        for b in range(B):
            pos = q_masks_bool[b].nonzero(as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                q_list.append(inputs_embeds.new_zeros((0, D)))
            else:
                q_list.append(inputs_embeds[b, pos, :])
        q_padded, q_mask, q_lengths = self._pad_list_to_tensor(q_list, pad_value=0.0)
        return q_padded, q_mask, q_lengths, q_list
    
    def _pack_tokens(self, f_padded, f_mask, topk_mask):
        """
        Pack padded token sequences into concatenated form.
        Args:
            f_padded: [B, T_max, D]
            f_mask:   [B, T_max], 1 = valid, 0 = pad
            topk_mask: [B, T_max], top-K binary mask

        Returns:
            f_packed: [sum(T), D], concatenated tokens
            topk_mask_packed: [sum(T)], concatenated top-K mask
        """
        B, T_max, D = f_padded.shape
        f_list, mask_list, topk_list = [], [], []

        for b in range(B):
            valid_len = f_mask[b].sum().item()
            f_list.append(f_padded[b, :valid_len])
            topk_list.append(topk_mask[b, :valid_len])

        f_packed = torch.cat(f_list, dim=0)
        topk_mask_packed = torch.cat(topk_list, dim=0)

        if f_packed.shape[0] != topk_mask_packed.shape[0]:
            raise RuntimeError("Packed features and top-K mask length mismatch")
        else:
            return f_packed, topk_mask_packed


    # --- VTS core -----------------------------------------------------------
    def _aggregate_per_frame(self, f_list, embeds_grid_tn, mode="mean"):
        """
        Aggregate n tokens within each frame to produce frame-level features.
        Args:
            f_list: list of [T_i, D] (all visual tokens per sample)
            embeds_grid_tn: [B, 2], each row is (T_frame_i, n_i)
        Returns:
            frame_list: list of [T_frame_i, D] (frame-level aggregated features)
        """
        frame_list = []
        B = len(f_list)
        for b in range(B):
            f_b = f_list[b]
            if f_b.shape[0] == 0:
                frame_list.append(f_b)
                continue
            T_frame, n = embeds_grid_tn[b].tolist()
            if T_frame * n != f_b.shape[0]:
                T_frame = f_b.shape[0] // n
            f_b = f_b.view(T_frame, n, -1)
            if mode == "mean":
                f_frame = f_b.mean(dim=1)
            elif mode == "max":
                f_frame, _ = f_b.max(dim=1)
            else:
                raise ValueError(f"Unknown mode {mode}")
            frame_list.append(f_frame)
        return frame_list

    def pool_query(self, q, mask=None):
        # q: [B, L, D_q], mask: [B, L] or [B, L, 1], 1=valid, 0=pad
        if not self.use_att_pool:
            if mask is None:
                return q.mean(dim=1, keepdim=True)

            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)

            q_masked = q * mask
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            q_avg_pooled = q_masked.sum(dim=1, keepdim=True) / denom
            return q_avg_pooled
        else:
            a = self.q_pool_att(q).squeeze(-1)
            if mask is not None:
                a = a.masked_fill(mask.squeeze(-1) == 0, -1e9)
            a = torch.softmax(a, dim=1)
            return torch.einsum("bl, bld -> b1d", a, q)

    def forward(self, video_embeds, inputs_embeds, input_ids, embeds_grid_tn=None):
        """
        Main entry.
        Args:
            video_embeds: [N_vis, D_vis]
            inputs_embeds: [B, T_total, D_vis]
            input_ids: [B, T_total]
        Returns:
            video_embeds_pruned: [N_vis, D_vis]
            video_topk_mask_packed: [N_vis]      (1 selected, 0 not)
            (optional) inputs_embeds_with_pruned_vis: inputs_embeds with visual token embeddings replaced
        """
        device = video_embeds.device
        B, T_total, D = inputs_embeds.shape

        f_list, counts = self._split_video_embeds_by_input_ids(video_embeds, input_ids)
        
        # Frame-level aggregation branch
        if embeds_grid_tn is not None:
            f_frame_list = self._aggregate_per_frame(f_list, embeds_grid_tn, mode="mean")
            f_padded, f_mask, f_lengths = self._pad_list_to_tensor(f_frame_list, pad_value=0.0)
        else:
            f_padded, f_mask, f_lengths = self._pad_list_to_tensor(f_list, pad_value=0.0)

        f_mask_float = f_mask.to(video_embeds.dtype)

        q_padded, q_mask, q_lengths, q_list = self._extract_queries_from_inputs(inputs_embeds, input_ids)
        q_pool = self.pool_query(q_padded, q_mask)

        B, T_max, D = f_padded.shape
        if not self.parameters_free:
            f_padded = f_padded.to(self.Wf.weight.dtype)
            q_pool = q_pool.to(self.Wq.weight.dtype)
        else:
            f_padded = f_padded.to(video_embeds.dtype)
            q_pool = q_pool.to(video_embeds.dtype)

        if self.parameters_free:
            f_norm = F.normalize(f_padded, dim=-1)
            q_norm = F.normalize(q_pool, dim=-1)

            scores = torch.sum(f_norm * q_norm, dim=-1)
            scores = scores.masked_fill(f_mask == 0, float('-inf'))
        else:
            f_proj = self.Wf(f_padded)
            q_proj = self.Wq(q_pool).squeeze(1)

            scores = torch.einsum('btd,bd->bt', f_proj, q_proj)
            scores = scores.masked_fill(f_mask == 0, float('-inf'))

        # Expand frame-level scores back to token-level
        if embeds_grid_tn is not None:
            T_token_max = int((embeds_grid_tn[:, 0] * embeds_grid_tn[:, 1]).max().item())
            f_padded_token = f_padded.new_zeros((B, T_token_max, D))
            f_mask_token = torch.zeros((B, T_token_max), dtype=torch.long, device=f_padded.device)
            scores_token = scores.new_full((B, T_token_max), float('-inf'))
            for b in range(B):
                T_frame, n = embeds_grid_tn[b].tolist()
                T_frame = int(T_frame)
                n = int(n)
                if T_frame == 0:
                    continue

                frames_b = f_padded[b, :T_frame, :]
                frames_rep = frames_b.repeat_interleave(n, dim=0)
                f_padded_token[b, :frames_rep.shape[0], :] = frames_rep
                
                s = scores[b, :T_frame]
                s_rep = s.repeat_interleave(n)
                scores_token[b, :s_rep.shape[0]] = s_rep
                f_mask_token[b, :frames_rep.shape[0]] = 1
            
            f_padded = f_padded_token
            f_mask = f_mask_token
            f_mask_float = f_mask.to(video_embeds.dtype)
            scores = scores_token

        soft_w = torch.softmax(scores / self.temp, dim=1) * f_mask_float

        valid_counts = f_mask.sum(dim=1)
        K_per_sample = torch.clamp((valid_counts.float() * self.token_ratio).ceil().long(), min=0)
        K_per_sample = torch.where(valid_counts > 0, torch.clamp(K_per_sample, min=1), torch.zeros_like(K_per_sample))

        new_f_padded = f_padded.clone()
        topk_mask_padded = torch.zeros_like(f_mask, dtype=torch.long)

        if self.batch_topk:
            K_max = int(K_per_sample.max().item())
            topk_vals, topk_idx = torch.topk(soft_w, k=K_max, dim=1)

            # Per-sample k mask: only keep top-K entries per sample
            arange_k = torch.arange(K_max, device=K_per_sample.device).unsqueeze(0)
            k_mask = arange_k < K_per_sample.unsqueeze(1)

            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, D)
            f_sel = torch.gather(f_padded, 1, idx_exp)

            # Mask out entries beyond k per sample before softmax
            w_sel = torch.softmax(
                (topk_vals / self.temp).masked_fill(~k_mask, -1e9), dim=1
            ).unsqueeze(-1)

            if self.post_mlp is not None:
                f_sel_proc = self.post_mlp(f_sel) * w_sel
            else:
                f_sel_proc = f_sel * w_sel

            batch_idx = torch.arange(B, device=f_padded.device).unsqueeze(1).expand_as(topk_idx)
            new_f_padded = new_f_padded.index_put(
                (batch_idx, topk_idx),
                f_sel_proc,
                accumulate=False
            )

            # Clear excess topk marks beyond k per sample
            valid_topk_idx = topk_idx.masked_fill(~k_mask, 0)
            batch_idx = torch.arange(B, device=f_mask.device).unsqueeze(1).expand_as(valid_topk_idx)
            topk_mask_padded[batch_idx, valid_topk_idx] = k_mask.long()
            topk_mask_padded = topk_mask_padded * f_mask
        
        else:
            for b in range(B):
                t_len = int(f_lengths[b])
                if t_len == 0:
                    continue
                k = int(K_per_sample[b].item())
                if k == 0:
                    continue
                scores_b = scores[b, :t_len]
                soft_w_b = soft_w[b, :t_len]
                f_b = f_padded[b, :t_len]
                f_proj_b = f_proj[b, :t_len]

                topk = torch.topk(soft_w_b, k=k, dim=0)
                idx = topk.indices
                mask_b = torch.zeros((t_len,), device=device, dtype=torch.long)
                mask_b[idx] = 1
                topk_mask_padded[b, :t_len] = mask_b

                f_sel = f_b[idx]
                w_sel = torch.softmax(soft_w_b[idx] / self.temp, dim=0)
                w_sel = w_sel.unsqueeze(-1)

                if self.post_mlp is not None:
                    f_sel_proc = self.post_mlp(f_sel) * w_sel
                else:
                    f_sel_proc = f_sel * w_sel

                new_f_padded[b, :t_len][idx] = f_sel_proc
                
        f_packed, topk_mask_packed = self._pack_tokens(new_f_padded, f_mask, topk_mask_padded)

        return f_packed, topk_mask_packed, soft_w
