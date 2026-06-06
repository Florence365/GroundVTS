from collections import OrderedDict

import nncore
from torch.utils.data import Dataset

from eval.dataset.registry import DATASETS
from eval.utils.parser import parse_query


@DATASETS.register(name='youcook2')
class YouCook2Dataset(Dataset):

    ANNO_PATH = 'data/youcook2/youcookii_annotations_trainval.json'

    VIDEO_ROOT = 'data/youcook2/videos_3fps_480_noaudio'

    UNIT = 1.0

    @classmethod
    def load_annos(self, split='train'):
        subset = 'training' if split == 'train' else 'validation'

        raw_annos = nncore.load(self.ANNO_PATH, object_pairs_hook=OrderedDict)['database']

        all_videos = nncore.ls(self.VIDEO_ROOT, ext='.mp4')
        all_videos = set(v[:11] for v in all_videos)

        annos = []
        for vid, raw_anno in raw_annos.items():
            if raw_anno['subset'] != subset:
                continue

            if vid not in all_videos:
                continue

            duration = raw_anno['duration']

            for meta in raw_anno['annotations']:
                anno = dict(
                    source='youcook2',
                    data_type='grounding',
                    video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                    duration=duration,
                    query=parse_query(meta['sentence']),
                    span=[meta['segment']])

                annos.append(anno)

            annos.append(anno)

        return annos