import argparse
import copy
import json
from contextlib import nullcontext

import nncore
import torch
import time

from qwen_vl_utils import process_vision_info
from .dataset.registry import DATASETS
from .utils.io import get_duration


def parse_model_timestamp_response(response, duration=None):
    """
    Parse timestamp range from model response. Supports multiple formats:
    1. Two numbers like "11 to 15 seconds" or "11.0 to 15.2 sec"
    2. Multiple numbers: extract the first two
    3. Single number: infer start/end based on positional keywords
    Returns (start, end) as floats, or (None, None) on failure.
    """
    import re
    
    pattern = r'(\d+(?:\.\d+)?)\s*(?:s|seconds|sec)?\s*(?:and|to|-|–|—)\s*(\d+(?:\.\d+)?)\s*(?:s|seconds|sec)?'
    match = re.search(pattern, response.lower())
    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        return start, end
    
    all_numbers = re.findall(r'\d+(?:\.\d+)?', response)
    
    if len(all_numbers) >= 2:
        start = float(all_numbers[0])
        end = float(all_numbers[1])
        return start, end
    
    elif len(all_numbers) == 1 and duration is not None:
        single_num = float(all_numbers[0])
        response_lower = response.lower()
        
        beginning_keywords = [
            'beginning', 'start', 'first', 'opening', 'early', 'initial',
            'at the start', 'at the beginning', 'from the start'
        ]
        
        ending_keywords = [
            'end', 'last', 'final', 'closing', 'late', 'toward the end',
            'at the end', 'near the end', 'towards the end'
        ]
        
        has_beginning = any(keyword in response_lower for keyword in beginning_keywords)
        has_ending = any(keyword in response_lower for keyword in ending_keywords)
        
        if has_beginning and not has_ending:
            start = 0
            end = min(single_num, duration)
            return start, end
        
        elif has_ending and not has_beginning:
            start = max(0, duration - single_num) 
            end = duration
            return start, end
        
        return None, None
    
    return None, None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='intern')
    parser.add_argument('--dataset', default='charades_sta')    
    parser.add_argument('--pred_path', default='<path/to/predictions>')
    parser.add_argument('--vts_ratio', default=0.5, type=float)
    parser.add_argument('--base_model_path', default='<path/to/base_model>')
    parser.add_argument('--lora_adapter_paths', default=None)
    parser.add_argument('--fps', default=1.0, type=float)
    parser.add_argument('--num_frames', default=8, type=int)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
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
        pred_path = nncore.join(args.pred_path, f'output_{args.dataset}_{args.split}_{args.fps}_{args.num_frames}_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, f'output_{args.dataset}_{args.split}_{args.fps}_{args.num_frames}.json')

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

        video_path, duration, span = anno['video_path'], anno.get('duration'), anno.get('span')

        do_answering = all(k in anno for k in ('question', 'options'))
        do_grounding = anno['data_type']=='grounding'

        if not do_grounding and do_answering:
            question, options, ans = anno['question'], anno['options'], anno['ans']

            if args.style in ('mcq', 'options') and len(options) > 0:
                prompt = question + '\nOptions:'
                for idx, opt in enumerate(options):
                    if len(opt) == 0:
                        continue
                    prompt += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
                prompt += '\nPlease only give the best option.'
            else:
                prompt = question

        elif do_grounding:
            question = anno['query']
            prompt = f"Based on the video, please provide the start and end timestamps (in seconds) for the moment described by: '{question}'. Respond in the format 'X to Y seconds'. If unsure, make your best guess."
        else:
            question = anno['query']
            prompt = question

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
            if args.dataset == 'mvbench':
                do_sample_frames = False
            else:
                do_sample_frames = True
            try:
                inputs = processor.apply_chat_template(messages,
                                                    return_tensors="pt",
                                                    add_generation_prompt=True,
                                                    tokenize=True,
                                                    return_dict=True,
                                                    num_frames=args.num_frames,
                                                    do_sample_frames=do_sample_frames,
                                                ).to(model.device, dtype=torch.float16)
            except Exception as e:
                print("Video loading failed")
                continue
            output = model.generate(**inputs, max_new_tokens=256, use_cache=False)
            response = processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        dump['response'] = response

        if do_grounding:
            if duration is None:
                duration = get_duration(video_path, num_threads=args.num_threads)
                dump['duration'] = duration

            pre_start, pre_end = parse_model_timestamp_response(response, duration=duration)
            dump['pred'] = [pre_start, pre_end]
            
        
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
