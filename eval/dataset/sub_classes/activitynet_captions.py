from collections import OrderedDict

import nncore
from torch.utils.data import Dataset

from eval.dataset.registry import DATASETS
from eval.utils.parser import parse_query


@DATASETS.register(name='activitynet_captions')
class ActivitynetCaptionsDataset(Dataset):

    ANNO_PATH_TRAIN = '<path/to/activitynet_captions_train.json>'
    ANNO_PATH_VALID = '<path/to/activitynet_captions_val1.json>'
    ANNO_PATH_TEST = '<path/to/activitynet_captions_val2.json>'

    VIDEO_ROOT = '<path/to/activitynet_videos>'
    DURATIONS = '<path/to/activitynet_durations.json>'

    UNIT = 0.01

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN, object_pairs_hook=OrderedDict)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID, object_pairs_hook=OrderedDict)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST, object_pairs_hook=OrderedDict)

        durations = nncore.load(self.DURATIONS)

        annos = []
        iterable = [(anno.get("vid") or anno.get("video_id"), anno) for anno in raw_annos]
        for vid, raw_anno in iterable:
            for query, span in zip(raw_anno['sentences'], raw_anno['timestamps']):
                try:
                    anno = dict(
                        source='activitynet_captions',
                        data_type='grounding',
                        video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                        duration=durations[vid],
                        query=parse_query(query),
                        span=[span])

                    annos.append(anno)
                except KeyError:
                    print(f'Warning: video {vid} not found in durations.json')
                    continue

        return annos
