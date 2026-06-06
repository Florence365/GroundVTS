import os
import shutil
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import argparse
from models.vts_qwen2_5_vl.modeling_vts_qwen import VTS_Qwen2_5_VL, VTS_Qwen2_5_VLConfig

def safe_copy(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"Copied: {os.path.basename(src)}")

def copy_configs(base_dir, lora_dir, out_dir):
    """
    Copy config files from base model to out_dir,
    then overwrite with any updated configs from the LoRA directory.
    """
    print("\nCopying base + LoRA config files...")

    for f in os.listdir(base_dir):
        src = os.path.join(base_dir, f)
        dst = os.path.join(out_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    overwrite_files = [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "processor_config.json",
        "vocab.json",
        "merges.txt",
    ]
    for f in overwrite_files:
        src = os.path.join(lora_dir, f)
        dst = os.path.join(out_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Overwrite {f} from LoRA adapter")

    print("Config copy complete.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default='<path/to/base_model>', help="Path to base model")
    parser.add_argument("--lora", default='<path/to/lora_adapter>', help="Path to LoRA adapter")
    parser.add_argument("--out", default='<path/to/merged_output>', help="Output path for merged model")
    parser.add_argument("--device", default="cuda", help="cpu or cuda:0")
    args = parser.parse_args()

    device = args.device

    print(f"Loading base model from {args.base} to {device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map=None, torch_dtype=torch.float32
    )

    print(f"Loading LoRA adapter from {args.lora}")
    model = PeftModel.from_pretrained(model, args.lora, torch_device=device)

    print("Merging LoRA weights into base model ...")
    model = model.merge_and_unload()
    model._hf_peft_config_loaded = False

    print(f"Saving merged model to {args.out}")
    model.save_pretrained(args.out)

    copy_configs(args.base, args.lora, args.out)

    print("All done! Merged model ready at:", args.out)


if __name__ == "__main__":
    main()
