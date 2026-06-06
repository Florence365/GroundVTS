import warnings
import nncore
import torch
from torch import nn
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, GenerationConfig
from .modeling_vts_intern import VTS_InternVL_3, VTS_InternVL_3Config


def get_auto_device():
    """Select device: cuda > npu > cpu"""
    try:
        import torch_npu
        has_npu = torch_npu.npu.is_available()
    except ImportError:
        has_npu = False

    return 'cuda' if torch.cuda.is_available() else 'npu' if has_npu else 'cpu'


def build_model(
    base_model_path,
    lora_adapter_paths=None,   # None / str / list[str]
    is_trainable=False,
    merge_adapter=True,
    device="auto",
    dtype="auto"
):
    """
    Build model with optional multi-LoRA adapter stacking.
    """
    processor = AutoProcessor.from_pretrained(base_model_path)
    config = AutoConfig.from_pretrained(base_model_path)

    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=dtype,
        device_map="auto" if device == "all" else None,
        low_cpu_mem_usage=True
    )

    if lora_adapter_paths is not None:
        if isinstance(lora_adapter_paths, str):
            lora_adapter_paths = [lora_adapter_paths]

        for idx, lora_path in enumerate(lora_adapter_paths):
            if nncore.is_dir(lora_path):
                print(f"Loading LoRA adapter {idx+1}: {lora_path} ...")
                model = PeftModel.from_pretrained(
                    model,
                    lora_path,
                    is_trainable=is_trainable,
                    torch_device=str(model.get_input_embeddings().weight.device)
                )
        
        if merge_adapter:
            print("Merging all LoRA adapters and unloading...")
            model = model.merge_and_unload()
            model._hf_peft_config_loaded = False

    if not is_trainable and device != "all":
        device = get_auto_device() if device == "auto" else device
        model = model.to(device).eval()

    return model, processor
