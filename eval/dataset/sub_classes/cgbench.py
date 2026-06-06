import nncore
from torch.utils.data import Dataset

from eval.dataset.registry import DATASETS
from eval.utils.parser import parse_query, parse_question


@DATASETS.register(name='cgbench')
class CGBenchDataset(Dataset):

    ANNO_PATH_TEST = '<path/to/cgbench>/cgbench_mini.json'

    VIDEO_ROOT = '<path/to/cgbench>/video_chunk'
    DURATIONS = '<path/to/cgbench>/durations.json'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='test'):
        assert split == 'test'

        raw_annos = nncore.load(self.ANNO_PATH_TEST)

        durations = nncore.load(self.DURATIONS)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['video_uid']

            anno = dict(
                source='cgbench',
                data_type='multimodal',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=durations[vid],
                query=parse_query(raw_anno['question']),
                question=parse_question(raw_anno['question']),
                options=[o[0].upper() + o[1:] for o in raw_anno['choices']],
                answer=raw_anno['answer'][0].upper() + raw_anno['answer'][1:],
                ans=raw_anno['right_answer'],
                span=raw_anno['clue_intervals'],
                task=raw_anno['sub_category'],
                domain=raw_anno['domain'])

            annos.append(anno)

        return annos
