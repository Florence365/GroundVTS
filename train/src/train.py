# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from llamafactory.train.tuner import run_exp

try:
    from models.vts_qwen2_5_vl.modeling_vts_qwen import VTS_Qwen2_5_VL, VTS_Qwen2_5_VLConfig
    from models.vts_internvl_3.modeling_vts_intern import VTS_InternVL_3, VTS_InternVL_3Config
except ImportError as e:
    print(f"Failed to import custom model: {e}")
    sys.exit(1)


def main():
    run_exp()


def _mp_fn(index):
    run_exp()


if __name__ == "__main__":
    main()
