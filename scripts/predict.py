import json

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from common import load_config, load_inference_model, shared_parser
from metrics import parse_predictions


def main():
    parser = shared_parser("Run object detection on one image")
    parser.add_argument("image")
    parser.add_argument("--adapter-path", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    adapter_path = args.adapter_path or f"{args.output_dir}/best_model"
    model, processor = load_inference_model(config, args.model_path, adapter_path)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": config["prompt"]}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=config.get("max_new_tokens", 512), do_sample=False)
    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(json.dumps(parse_predictions(response, config["class_names"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
