import inspect
import json
import logging
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import AutoProcessor, EarlyStoppingCallback, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from common import dataset_paths, load_config, require_paths, resolve_model_source, set_seed, shared_parser


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DetectionDataset(Dataset):
    def __init__(self, annotation_path, image_dir):
        self.image_dir = image_dir
        with open(annotation_path, "r", encoding="utf-8") as file:
            self.samples = json.load(file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = os.path.join(self.image_dir, sample["image"])
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        human_text = sample["conversations"][0]["value"].replace("<image>\n", "", 1)
        assistant_text = sample["conversations"][1]["value"]
        return {
            "user": {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": human_text}]},
            "assistant": {"role": "assistant", "content": assistant_text},
        }


class DetectionCollator:
    def __init__(self, processor, max_length):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        conversations = [[item["user"], item["assistant"]] for item in batch]
        prompts = [[item["user"]] for item in batch]
        texts = [self.processor.apply_chat_template(item, tokenize=False, add_generation_prompt=False) for item in conversations]
        prompt_texts = [self.processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in prompts]
        images, videos = process_vision_info(conversations)
        inputs = self.processor(text=texts, images=images, videos=videos, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt_texts, images=images, videos=videos, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        # 屏蔽图片和用户指令，只对 assistant 的检测结果计算损失。
        for row in range(labels.size(0)):
            prompt_length = int(prompt_inputs["attention_mask"][row].sum())
            labels[row, :prompt_length] = -100
        inputs["labels"] = labels
        return {key: value for key, value in inputs.items() if isinstance(value, torch.Tensor)}


def main():
    parser = shared_parser("Fine-tune Qwen2.5-VL with LoRA for object detection")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    training = dict(config["training"])
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    paths = dataset_paths(args.data_dir)
    require_paths(paths, ("train_json", "validation_json", "train_images", "validation_images"))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    set_seed(training["seed"])
    os.makedirs(args.output_dir, exist_ok=True)
    model_source = resolve_model_source(config, args.model_path)
    processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True, use_fast=False)
    processor.tokenizer.padding_side = "right"
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16 if training["bf16"] else torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()
    lora = config["lora"]
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora["target_modules"],
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            bias=lora["bias"],
            inference_mode=False,
        ),
    )
    model.print_trainable_parameters()
    kwargs = {
        "output_dir": args.output_dir,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "max_steps": training["max_steps"],
        "per_device_train_batch_size": training["train_batch_size"],
        "per_device_eval_batch_size": training["eval_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "warmup_steps": training["warmup_steps"],
        "lr_scheduler_type": training["lr_scheduler_type"],
        "eval_strategy": "steps",
        "eval_steps": training["eval_steps"],
        "save_strategy": "steps",
        "save_steps": training["save_steps"],
        "save_total_limit": training["save_total_limit"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": training["logging_steps"],
        "report_to": "tensorboard",
        "bf16": training["bf16"],
        "fp16": training["fp16"],
        "gradient_checkpointing": training["gradient_checkpointing"],
        "dataloader_num_workers": training["dataloader_num_workers"],
        "remove_unused_columns": False,
        "seed": training["seed"],
        "data_seed": training["seed"],
    }
    supported = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    kwargs = {key: value for key, value in kwargs.items() if key in supported}
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**kwargs),
        train_dataset=DetectionDataset(paths["train_json"], paths["train_images"]),
        eval_dataset=DetectionDataset(paths["validation_json"], paths["validation_images"]),
        data_collator=DetectionCollator(processor, training["model_max_length"]),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training["early_stopping_patience"])],
    )
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as file:
        json.dump({"config": config, "data_dir": os.path.abspath(args.data_dir), "model_source": model_source}, file, indent=2, ensure_ascii=False)
    # 默认从最新完整 checkpoint 续训，best_model 仅用于最终推理。
    checkpoint = None if args.no_resume else get_last_checkpoint(args.output_dir)
    logger.info("Resume checkpoint: %s", checkpoint or "none")
    result = trainer.train(resume_from_checkpoint=checkpoint)
    best_dir = os.path.join(args.output_dir, "best_model")
    trainer.model.save_pretrained(best_dir)
    processor.save_pretrained(best_dir)
    trainer.save_state()
    metadata = {
        "selection_metric": "eval_loss",
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "stopped_step": trainer.state.global_step,
    }
    with open(os.path.join(best_dir, "selection.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    with open(os.path.join(args.output_dir, "training_results.json"), "w", encoding="utf-8") as file:
        json.dump(result.metrics, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
