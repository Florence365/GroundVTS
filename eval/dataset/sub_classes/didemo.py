# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import random
from torch.utils.data import Dataset
import nncore
import numpy as np

from eval.dataset.registry import DATASETS
from eval.utils.parser import parse_query


@DATASETS.register(name='didemo')
class DiDeMoDataset(Dataset):

    ANNO_PATH_TRAIN = '<path/to/didemo>/didemo_train.json'
    ANNO_PATH_VALID = '<path/to/didemo>/didemo_val.json'
    ANNO_PATH_TEST = '<path/to/didemo>/didemo_test.json'

    VIDEO_ROOT = '<path/to/didemo>/video/test'
    DURATIONS = '<path/to/didemo>/durations.json'

    UNIT = 1.0

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        durations = nncore.load(self.DURATIONS)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video'].split('.')[0]

            # apply mean on multiple spans
            span = np.array(raw_anno['times']).mean(axis=0).tolist()
            span = [round(span[0] * 5), round((span[1] + 1) * 5)]

            # augment spans during training
            if split == 'train':
                offset = random.randint(-2, 2)
                span = [span[0] + offset, span[1] + offset]

            anno = dict(
                source='didemo',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=durations[vid],
                query=parse_query(raw_anno['description']),
                span=[span])

            annos.append(anno)

        return annos
