import argparse

import nncore
import torch
from nncore.ops import temporal_area, temporal_intersection, temporal_iof, temporal_iou
from tabulate import tabulate


class SafeInt(int):

    def __truediv__(self, other):
        try:
            return SafeInt(super().__truediv__(other))
        except ZeroDivisionError:
            return SafeInt(0)


def check_ans(options, ans, response):
    a = ans.lower()
    response_lower = response.lower()
    
    import re
    patterns = [
        r'[\(\[\{]\s*([a-d])\s*[\)\]\}]',
        r'\b([a-d])[\s\.\,\)\]\}]',
        r'option\s*([a-d])',
        r'answer\s*is\s*([a-d])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            b = match.group(1)
            if b in [chr(ord('a') + i) for i in range(len(options))]:
                return a == b
    
    # Fallback to original parsing logic if regex matching fails
    b = response_lower.split(' ')[0].replace('(', '').replace(')', '').replace('.', '')
    if not b:
        nncore.log(f'ERROR: Empty response -> "{response}"')
        return False

    if len(b) > 1:
        nncore.log(f'WARNING: Unexpected answer format: "{response}" -> using first char "{b[0]}"')
        b = b[0]

    if b not in [chr(ord('a') + i) for i in range(len(options))]:
        nncore.log(f'ERROR: Invalid option in response: "{response}" -> "{b}"')
        return False

    return a == b

def compute_iou(pred, span, conf=None, cgbench_mode=False, conf_thr=-1):
    if pred is None or span is None:
        return torch.tensor([0.0])
    print(f'Pred: {pred}, Span: {span}, Conf: {conf}')
    try:
        pred_tensor = torch.as_tensor(pred, dtype=torch.float32).reshape(-1, 2)
        span_tensor = torch.as_tensor(span, dtype=torch.float32).reshape(-1, 2)
        if cgbench_mode and conf is not None:
            if  conf_thr > 0:
                conf_tensor = torch.Tensor(conf)
                keep = torch.cat((torch.LongTensor([0]), torch.where(conf_tensor > conf_thr)[0])).unique()
                pred_tensor = pred_tensor[keep]
            else:
                pred_tensor = pred_tensor[:1]
            pred_area = temporal_area(pred_tensor).sum()
            span_area = temporal_area(span_tensor).sum()
            inter = temporal_intersection(pred_tensor, span_tensor).sum()
            iou = (inter / (pred_area + span_area - inter)).unsqueeze(0)
            assert iou.numel() == 1
        else:
            iou = temporal_iou(pred_tensor, span_tensor)

        iou = torch.where(iou.isfinite(), iou, 0)
    except Exception as e:
        nncore.log(f'ERROR: {e}')
        iou = torch.tensor([0.0])
    return iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default='<path/to/predictions>')
    parser.add_argument('--pred_name', default='output')
    parser.add_argument('--if_cgbench', default=False)
    parser.add_argument('--conf_thr', type=float, default=-1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert nncore.is_dir(args.pred_path)

    log_file = nncore.join(args.pred_path, f"metrics_{args.pred_name}.log")
    nncore.set_default_logger(logger='eval', fmt=None, log_file=log_file)

    if args.if_cgbench is True:
        cgbench_mode = args.if_cgbench
        nncore.log(f'CG-Bench mode: {cgbench_mode}')
    else:
        cgbench_mode = False
        nncore.log('Dataset is unknown, using default mode', log_level='WARNING')

    pred_path = nncore.join(args.pred_path, f"{args.pred_name}.json")

    if cgbench_mode:
        top_k = [1]
        thres = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        top_k = [1]
        thres = [0.1, 0.3, 0.5, 0.7]

    tab_iou, tab_iop, tab_ans = dict(), dict(), dict()
    iou_raise, iou_lower, iop_raise, iop_lower = SafeInt(0), SafeInt(0), SafeInt(0), SafeInt(0)
    tab_iou_all = [SafeInt(0) for _ in range(len(top_k) * len(thres) + 3)]
    tab_iop_all = [SafeInt(0) for _ in range(len(top_k) * len(thres) + 3)]
    tab_ans_all = [SafeInt(0) for _ in range(len(thres) + 5)]

    data = nncore.load(pred_path)

    for sample in data:
        task = sample.get('task', 'unknown')

        if isinstance(task, str):
            task = [task]

        for t in task:
            if t not in tab_iou:
                tab_iou[t] = [SafeInt(0) for _ in range(len(top_k) * len(thres) + 3)]

            if t not in tab_iop:
                tab_iop[t] = [SafeInt(0) for _ in range(len(top_k) * len(thres) + 3)]

            if t not in tab_ans:
                tab_ans[t] = [SafeInt(0) for _ in range(len(thres) + 5)]

        iou_hit = [False for _ in range(len(thres) + 1)]
        iop_hit = False

        if 'pred' in sample and 'span' in sample:
            for t in task:
                tab_iou[t][0] += 1
                tab_iop[t][0] += 1
            tab_iou_all[0] += 1
            tab_iop_all[0] += 1

            if sample['pred'] is None or not isinstance(sample['pred'], (list, tuple)) or all(p is None for p in sample['pred']):
                tab_iou[t][1] += 1
                iou = torch.tensor([0.0])
            else:
                if 'conf' in sample:
                    iou = compute_iou(sample['pred'], sample['span'], sample['conf'], cgbench_mode, args.conf_thr)
                else:
                    iou = compute_iou(sample['pred'], sample['span'])
            top = iou[0].max().item()

            for t in task:
                tab_iou[t][-1] += top
            tab_iou_all[-1] += top

            for i, k in enumerate(top_k):
                for j, h in enumerate(thres):
                    if iou[:k].max() >= h:
                        for t in task:
                            tab_iou[t][i * len(thres) + j + 2] += 1
                        tab_iou_all[i * len(thres) + j + 2] += 1
                        if k == 1:
                            iou_hit[j + 1] = True
                            if h == 0.5:
                                iou_hit[0] = True
            
            if sample['pred'] is None or not isinstance(sample['pred'], (list, tuple)) or all(p is None for p in sample['pred']):
                tab_iop[t][1] += 1
                iop = torch.tensor([0.0])
            else:
                iop = temporal_iof(
                        torch.tensor(sample['pred'], dtype=torch.float32).reshape(-1, 2),
                        torch.tensor(sample['span'], dtype=torch.float32).reshape(-1, 2)
                    )

            iop = torch.where(iop.isfinite(), iop, 0)
            top = iop[0].max().item()

            for t in task:
                tab_iop[t][-1] += top
            tab_iop_all[-1] += top

            for i, k in enumerate(top_k):
                for j, h in enumerate(thres):
                    if iop[:k].max() >= h:
                        for t in task:
                            tab_iop[t][i * len(thres) + j + 2] += 1
                        tab_iop_all[i * len(thres) + j + 2] += 1
                        if k == 1 and h == 0.5:
                            iop_hit = True

            if sample['pred'] is None:
                for t in task:
                    tab_iou[t][1] += 1
                    tab_iop[t][1] += 1
                tab_iou_all[1] += 1
                tab_iop_all[1] += 1

        if 'question' in sample and 'response' in sample:
            for t in task:
                tab_ans[t][0] += 1
            tab_ans_all[0] += 1

            if 'answer' not in sample:
                continue
            else:
                correct = check_ans(sample['options'], sample['ans'], sample['response'])

            if correct:
                for t in task:
                    tab_ans[t][2] += 1
                tab_ans_all[2] += 1
                if iou_hit[0]:
                    for t in task:
                        tab_ans[t][3] += 1
                    tab_ans_all[3] += 1
                if iop_hit:
                    for t in task:
                        tab_ans[t][4] += 1
                    tab_ans_all[4] += 1
                for i in range(1, len(iou_hit)):
                    if iou_hit[i]:
                        for t in task:
                            tab_ans[t][i + 4] += 1
                        tab_ans_all[i + 4] += 1
            elif correct is None:
                for t in task:
                    tab_ans[t][1] += 1
                tab_ans_all[1] += 1

    tasks = sorted(list(set(list(tab_iou.keys()) + list(tab_iop.keys()) + list(tab_ans.keys()))))

    if cgbench_mode:
        nncore.log('\nGrounding (IoU):')
        tab = tabulate(
            [[task, tab_iou[task][0], tab_iou[task][1]] +
             [f'{tab_iou[task][i] / tab_iou[task][0] * 100:.2f}' for i in range(2, len(tab_iou[task]))] +
             [f'{sum(tab_iou[task][i] / tab_iou[task][0] for i in range(2, 2 + len(thres))) / len(thres) * 100:.2f}']
             for task in tasks if task in tab_iou] +
            [['all', tab_iou_all[0], tab_iou_all[1]] +
             [f'{tab_iou_all[i] / tab_iou_all[0] * 100:.2f}' for i in range(2, len(tab_iou_all))] +
             [f'{sum(tab_iou_all[i] / tab_iou_all[0] for i in range(2, 2 + len(thres))) / len(thres) * 100:.2f}']],
            headers=['Task', '#Samples', 'Failed'] + [f'R{k}@{t}' for k in top_k for t in thres] + ['mIoU', 'rec.@IoU'],
            tablefmt='pretty',
            stralign='left')
        nncore.log(tab)

        nncore.log(f'\nIoU Raise ({tab_iou_all[0]} Samples): {iou_raise} ({iou_raise / tab_iou_all[0] * 100:.2f}%)')
        nncore.log(f'IoU Lower ({tab_iou_all[0]} Samples): {iou_lower} ({iou_lower / tab_iou_all[0] * 100:.2f}%)')

        nncore.log('\nQA:')
        tab = tabulate(
            [[task, tab_ans[task][0], tab_ans[task][1], f'{tab_ans[task][2] / tab_ans[task][0] * 100:.2f}'] +
             [f'{sum(tab_ans[task][i] / tab_ans[task][0] for i in range(5, 5 + len(thres))) / len(thres) * 100:.2f}']
             for task in tasks if task in tab_ans] +
            [['all', tab_ans_all[0], tab_ans_all[1], f'{tab_ans_all[2] / tab_ans_all[0] * 100:.2f}'] +
             [f'{sum(tab_ans_all[i] / tab_ans_all[0] for i in range(5, 5 + len(thres))) / len(thres) * 100:.2f}']],
            headers=['Task', '#Samples', 'Failed', 'long-acc.', 'acc.@IoU'],
            tablefmt='pretty',
            stralign='left')
        nncore.log(tab)
    else:
        nncore.log('\nGrounding (IoU):')
        tab = tabulate(
            [[task, tab_iou[task][0], tab_iou[task][1]] +
             [f'{tab_iou[task][i] / tab_iou[task][0] * 100:.2f}' for i in range(2, len(tab_iou[task]))]
             for task in tasks if task in tab_iou] +
            [['all', tab_iou_all[0], tab_iou_all[1]] +
             [f'{tab_iou_all[i] / tab_iou_all[0] * 100:.2f}' for i in range(2, len(tab_iou_all))]],
            headers=['Task', '#Samples', 'Failed'] + [f'R{k}@{t}' for k in top_k for t in thres] + ['mIoU'],
            tablefmt='pretty',
            stralign='left')
        nncore.log(tab)

        nncore.log(f'\nIoU Raise ({tab_iou_all[0]} Samples): {iou_raise} ({iou_raise / tab_iou_all[0] * 100:.2f}%)')
        nncore.log(f'IoU Lower ({tab_iou_all[0]} Samples): {iou_lower} ({iou_lower / tab_iou_all[0] * 100:.2f}%)')

        nncore.log('\nGrounding (IoP):')
        tab = tabulate(
            [[task, tab_iop[task][0], tab_iop[task][1]] +
             [f'{tab_iop[task][i] / tab_iop[task][0] * 100:.2f}' for i in range(2, len(tab_iop[task]))]
             for task in tasks if task in tab_iop] +
            [['all', tab_iop_all[0], tab_iop_all[1]] +
             [f'{tab_iop_all[i] / tab_iop_all[0] * 100:.2f}' for i in range(2, len(tab_iop_all))]],
            headers=['Task', '#Samples', 'Failed'] + [f'R{k}@{t}' for k in top_k for t in thres] + ['mIoP'],
            tablefmt='pretty',
            stralign='left')
        nncore.log(tab)

        nncore.log(f'\nIoP Raise ({tab_iop_all[0]} Samples): {iop_raise} ({iop_raise / tab_iop_all[0] * 100:.2f}%)')
        nncore.log(f'IoP Lower ({tab_iop_all[0]} Samples): {iop_lower} ({iop_lower / tab_iop_all[0] * 100:.2f}%)')

        nncore.log('\nQA:')
        tab = tabulate(
            [[task, tab_ans[task][0], tab_ans[task][1]] +
             [f'{tab_ans[task][i] / tab_ans[task][0] * 100:.2f}' for i in range(2, 5)]
             for task in tasks if task in tab_ans] +
            [['all', tab_ans_all[0], tab_ans_all[1]] +
             [f'{tab_ans_all[i] / tab_ans_all[0] * 100:.2f}' for i in range(2, 5)]],
            headers=['Task', '#Samples', 'Failed', 'Acc', 'Acc (IoU >= 0.5)', 'Acc (IoP >= 0.5)'],
            tablefmt='pretty',
            stralign='left')
        nncore.log(tab)
