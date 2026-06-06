from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers import Qwen2_5_VLConfig


class VTS_Qwen2_5_VLConfig(Qwen2_5_VLConfig):
    model_type = "vts_qwen2_5_vl" 

    def __init__(
        self,
        VTS_hidden_size=256,
        VTS_token_ratio=0.5,
        VTS_temp=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs) 

        self.VTS_hidden_size = VTS_hidden_size
        self.VTS_token_ratio = VTS_token_ratio
        self.VTS_temp = VTS_temp


CONFIG_MAPPING.register("vts_qwen2_5_vl", VTS_Qwen2_5_VLConfig)
