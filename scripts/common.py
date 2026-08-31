import argparse
import json
import os
import random

import numpy as np
import torch


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        config = json.load(file)
    required = ("model_id", "class_names", "prompt", "lora", "training")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")
    if not config["class_names"] or len(set(config["class_names"])) != len(config["class_names"]):
        raise ValueError("class_names must contain unique, non-empty values")
    return config


def dataset_paths(data_dir):
    return {
        "train_json": os.path.join(data_dir, "train.json"),
        "validation_json": os.path.join(data_dir, "validation.json"),
        "test_json": os.path.join(data_dir, "test.json"),
        "train_images": os.path.join(data_dir, "images", "train"),
        "validation_images": os.path.join(data_dir, "images", "validation"),
        "test_images": os.path.join(data_dir, "images", "test"),
    }


def require_paths(paths, keys):
    missing = [paths[key] for key in keys if not os.path.exists(paths[key])]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_shared_arguments(parser):
    parser.add_argument("--config", default="configs/example.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/qwen_lora")
    parser.add_argument("--model-path", default=None)
    return parser


def shared_parser(description):
    return add_shared_arguments(argparse.ArgumentParser(description=description))


def resolve_model_source(config, model_path):
    return model_path or config["model_id"]


def load_inference_model(config, model_path, adapter_path):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model_source = resolve_model_source(config, model_path)
    # 优先读取 adapter 中保存的 processor，旧 adapter 缺失时回退到基础模型。
    try:
        processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True, use_fast=False)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True, use_fast=False)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
    return model, processor
