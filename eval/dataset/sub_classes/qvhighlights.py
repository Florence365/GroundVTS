import nncore
from torch.utils.data import Dataset

from eval.dataset.registry import DATASETS
from eval.utils.parser import parse_query


@DATASETS.register(name='qvhighlights')
class QVHighlightsDataset(Dataset):

    ANNO_PATH_TRAIN = '<path/to/qvhighlights_train.jsonl>'
    ANNO_PATH_VALID = '<path/to/qvhighlights_val.jsonl>'
    ANNO_PATH_TEST = '<path/to/qvhighlights_test.jsonl>'

    VIDEO_ROOT = '<path/to/qvhighlights_videos>'

    UNIT = 2.0

    @classmethod
    def load_annos(self, split='train'):
        if split == 'train':
            raw_annos = nncore.load(self.ANNO_PATH_TRAIN)
        elif split == 'valid':
            raw_annos = nncore.load(self.ANNO_PATH_VALID)
        else:
            print('WARNING: Test split does not have ground truth annotations')
            raw_annos = nncore.load(self.ANNO_PATH_TEST)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['vid']
            qid = raw_anno['qid']

            anno = dict(
                source='qvhighlights',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=raw_anno['duration'],
                query=parse_query(raw_anno['query']),
                span=raw_anno.get('relevant_windows'),
                vid=vid,
                qid=qid)

            annos.append(anno)

        return annos
