import nncore
from torch.utils.data import Dataset

from eval.dataset.registry import DATASETS
from eval.utils.parser import parse_query


@DATASETS.register(name='charades_sta')
class CharadesSTADataset(Dataset):

    ANNO_PATH_TRAIN = '<path/to/charades_sta_train.txt>'
    ANNO_PATH_TEST = '<path/to/charades_sta_test.txt>'

    VIDEO_ROOT = '<path/to/charades_sta_videos>'
    DURATIONS = '<path/to/charades_sta_durations.json>'

    UNIT = 0.1

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        else:
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        durations = nncore.load(self.DURATIONS)

        annos = []
        for raw_anno in raw_annos:
            info, query = raw_anno.split('##')
            vid, s, e = info.split()

            anno = dict(
                source='charades_sta',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=durations[vid],
                query=parse_query(query),
                span=[[float(s), float(e)]])

            annos.append(anno)

        return annos
