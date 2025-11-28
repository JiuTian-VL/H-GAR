from typing import Dict
from h_gar.policy.base_image_policy import BaseImagePolicy


class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy) -> Dict:
        raise NotImplementedError()
