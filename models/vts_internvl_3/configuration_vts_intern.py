from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers import InternVLConfig

class VTS_InternVL_3Config(InternVLConfig):
    model_type = "vts_internvl_3" 

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


CONFIG_MAPPING.register("vts_internvl_3", VTS_InternVL_3Config)
