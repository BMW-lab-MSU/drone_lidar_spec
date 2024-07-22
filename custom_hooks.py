# proposal_logger_hook.py

import os
import pickle
from mmengine.hooks import Hook
from mmengine.runner import Runner

class ProposalLoggerHook(Hook):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch=None, outputs=None):
        proposals = outputs['proposals']
        scores = outputs['scores']
        img_id = data_batch['img_metas'][0].data['img_id']

        proposals_with_scores = {
            'img_id': img_id,
            'proposals': proposals,
            'scores': scores
        }

        output_path = os.path.join(self.output_dir, f'{img_id}_proposals.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(proposals_with_scores, f)
