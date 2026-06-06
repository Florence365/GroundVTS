
import argparse
import copy
import json
from contextlib import nullcontext
import re
import numpy as np

import nncore
import torch
import time

from qwen_vl_utils import process_vision_info
from .dataset.registry import DATASETS
from .utils.io import get_duration

HIGHLIGHT_LEVEL_MAP = {
    "very important": ["very important", "extremely important", "highly important"],
    "important": ["important", "moderately important", "medium importance"],
    "less important": ["less important", "not important", "low importance"]
}


def parse_model_highlight_response(response, level_map=None):
    """
    Extract saliency time intervals from model output, supporting natural language variants.
    Example input:
      "The highlights are: important from 100s to 110s, 112.0s-118.0s; very important: 50-60s"
    Returns:
      {'very important': [(50.0, 60.0)], 'important': [(100.0, 110.0), (112.0, 118.0)]}
    """
    if not response or not isinstance(response, str):
        return {}

    text = response.lower().strip()

    if level_map is None:
        level_map = HIGHLIGHT_LEVEL_MAP

    # Replace level keywords with tags to prevent nested matching (e.g. "very important" contains "important")
    for base, variants in level_map.items():
        tag = f"<{base.replace(' ', '_')}>"
        for v in sorted(variants, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(v)}\b", tag, text)

    result = {k: [] for k in level_map.keys()}

    for base in level_map.keys():
        tag = f"<{base.replace(' ', '_')}>"

        pattern = rf"{tag}[^<]*?(?:from|at|:)?([^<]+?)(?=<|$)"
        matches = re.findall(pattern, text)

        for m in matches:
            pairs = re.findall(
                r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)",
                m
            )
            single_times = re.findall(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", m)

            for s, e in pairs:
                result[base].append((float(s), float(e)))

            # Single time points default to 2s clips
            for s in single_times:
                start = float(s)
                result[base].append((start, start + 2.0))
    
    untagged_pairs = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)",
        text
    )
    untagged_single = re.findall(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)\b", text)

    already_tagged_times = {
        (round(s, 2), round(e, 2))
        for vals in result.values() for (s, e) in vals
    }

    for s, e in untagged_pairs:
        s, e = float(s), float(e)
        if (round(s, 2), round(e, 2)) not in already_tagged_times:
            result["important"].append((s, e))

    for s in untagged_single:
        start = float(s)
        seg = (start, start + 2.0)
        if seg not in already_tagged_times:
            result["important"].append(seg)

    cleaned = {}
    for k, v in result.items():
        if not v:
            continue
        uniq = sorted(list(set(v)), key=lambda x: x[0])
        cleaned[k] = uniq

    return cleaned


def map_highlight_response_to_scores(response, duration, clip_length=8):
    """
    Convert model highlight intervals into per-clip saliency scores.
    """
    frame_num = int(np.ceil(duration / clip_length))
    scores = [0.0] * frame_num

    parsed = parse_model_highlight_response(response)
    if not parsed:
        return scores

    level2score = {"very important": 4, "important": 3, "less important": 2}

    for level, intervals in parsed.items():
        val = level2score.get(level, 0.0)
        for start, end in intervals:
            start_i = int(start // clip_length)
            end_i = int(np.ceil(end / clip_length))
            for i in range(start_i, min(end_i, frame_num)):
                scores[i] = max(scores[i], val)

    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='qwen_qts')
    parser.add_argument('--dataset', default='qvhighlights')
    parser.add_argument('--pred_path', default='<path/to/predictions>')
    parser.add_argument('--vts_ratio', default=0.5, type=float)
    parser.add_argument('--base_model_path', default='<path/to/base_model>')
    parser.add_argument('--lora_adapter_paths', default=None)
    parser.add_argument('--clip_length',default=2)
    parser.add_argument('--fps', default=2.0, type=float)
    parser.add_argument('--num_frames', default=8, type=int)
    parser.add_argument('--split', default='valid', choices=['train', 'valid', 'test'])
    parser.add_argument('--style', default='mcq', choices=['mcq', 'options', 'direct'])
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--device', default='all', choices=['auto', 'all', 'cuda', 'npu', 'cpu'])
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    SAVE_INTERVAL = 20
    SAVE_TIME = 300
    last_save_time = time.time()

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.dataset}_{args.split}_{args.fps}_{args.num_frames}_{args.index}.jsonl')
    else:
        pred_path = nncore.join(args.pred_path, f'output_{args.dataset}_{args.split}_{args.fps}_{args.num_frames}.jsonl')

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')

    print('Initializing model')
    if args.model_type == 'qwen_qts':
        from models.vts_qwen2_5_vl.builder import build_model
        model, processor = build_model(
                base_model_path=args.base_model_path,
                lora_adapter_paths=args.lora_adapter_paths,
                merge_adapter=True,
                device=args.device,
                vts_ratio=args.vts_ratio
            )
    elif args.model_type == 'qwen':
        from models.qwen2_5_vl.builder import build_model
        model, processor = build_model(
                base_model_path=args.base_model_path,
                lora_adapter_paths=args.lora_adapter_paths,
                merge_adapter=True,
                device=args.device
            )
    elif args.model_type == 'intern':
        from models.internvl3_5.builder import build_model
        model, processor = build_model(
                base_model_path=args.base_model_path,
                lora_adapter_paths=args.lora_adapter_paths,
                merge_adapter=True,
                device=args.device
            )
    elif args.model_type == 'intern_qts':
        from models.vts_internvl_3.builder import build_model
        model, processor = build_model(
                base_model_path=args.base_model_path,
                lora_adapter_paths=args.lora_adapter_paths,
                merge_adapter=True,
                device=args.device
            )
    device = next(model.parameters()).device

    dataset_cls = DATASETS.get(args.dataset)
    annos = dataset_cls.load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = copy.deepcopy(annos[i])

        video_path, query, duration, span = anno['video_path'], anno['query'], anno.get('duration'), anno.get('span')

        prompt = f"Please highlight the most important parts for the following event: {query}"
        
        messages = [{
            'role': 'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'fps': args.fps,
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]

        if args.model_type == 'qwen' or args.model_type == 'qwen_qts':
            text = processor.apply_chat_template(messages, add_generation_prompt=True)

            try:
                images, videos = process_vision_info(messages)
                data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                data = data.to(device)
            except Exception as e:
                print(f'Error in processing vision info: {e}')
                continue

            with torch.inference_mode():
                output_ids = model.generate(
                    **data,
                    do_sample=False,
                    max_new_tokens=256)

            assert data.input_ids.size(0) == output_ids.size(0) == 1
            output_ids = output_ids[0, data.input_ids.size(1):]
            if output_ids[-1] == processor.tokenizer.eos_token_id:
                output_ids = output_ids[:-1]
            response = processor.decode(output_ids, clean_up_tokenization_spaces=False)
        elif args.model_type == 'intern' or args.model_type == 'intern_qts':
            inputs = processor.apply_chat_template(messages,
                                                    return_tensors="pt",
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_dict=True,
                                                    num_frames=args.num_frames,
                                                ).to(model.device, dtype=torch.float16)

            output = model.generate(**inputs, max_new_tokens=256, use_cache=False)
            response = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        pred_saliency_scores = map_highlight_response_to_scores(response, duration, clip_length=args.clip_length)
        dump['response'] = response
        dump["pred_saliency_scores"] = pred_saliency_scores
        
        dumps.append(dump)
        if (i + 1) % SAVE_INTERVAL == 0 or (time.time() - last_save_time) > SAVE_TIME:
            try:
                tmp_path = pred_path.replace(".json", f"_part_{args.index}.json")
                nncore.dump(dumps, tmp_path)
                print(f"[AutoSave] Saved {len(dumps)} results to {tmp_path}")
                last_save_time = time.time()
            except Exception as e:
                print(f"[AutoSave Error] Write failed: {e}")

    nncore.dump(dumps, pred_path)
