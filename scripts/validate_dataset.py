import json
import os
import sys

from PIL import Image

from common import dataset_paths, load_config, shared_parser


def validate_split(annotation_path, image_dir, class_names):
    errors = []
    with open(annotation_path, "r", encoding="utf-8") as file:
        samples = json.load(file)
    if not isinstance(samples, list):
        return [f"{annotation_path}: root must be a JSON array"], 0
    for index, sample in enumerate(samples):
        prefix = f"{annotation_path}[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{prefix}: sample must be an object")
            continue
        image_name = sample.get("image")
        conversations = sample.get("conversations")
        if not isinstance(image_name, str) or not image_name:
            errors.append(f"{prefix}: image must be a non-empty string")
            continue
        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            errors.append(f"{prefix}: image not found: {image_path}")
            continue
        if not isinstance(conversations, list) or len(conversations) != 2:
            errors.append(f"{prefix}: conversations must contain one human and one gpt message")
            continue
        human, assistant = conversations
        if not isinstance(human, dict) or human.get("from") != "human" or not isinstance(human.get("value"), str) or not human["value"].strip():
            errors.append(f"{prefix}: first conversation must be a non-empty human message")
            continue
        if not human["value"].startswith("<image>\n"):
            errors.append(f"{prefix}: human message must start with <image> followed by a newline")
        if not isinstance(assistant, dict) or assistant.get("from") != "gpt" or not isinstance(assistant.get("value"), str):
            errors.append(f"{prefix}: second conversation must be a gpt message containing JSON")
            continue
        # gpt.value 必须能独立解析为“类别 -> 边界框列表”的 JSON 对象。
        try:
            boxes = json.loads(assistant["value"])
        except json.JSONDecodeError as error:
            errors.append(f"{prefix}: gpt value is not valid JSON: {error}")
            continue
        if not isinstance(boxes, dict):
            errors.append(f"{prefix}: gpt JSON must be an object")
            continue
        unknown = sorted(set(boxes) - set(class_names))
        if unknown:
            errors.append(f"{prefix}: unknown classes: {unknown}")
        missing_classes = sorted(set(class_names) - set(boxes))
        if missing_classes:
            errors.append(f"{prefix}: missing classes: {missing_classes}")
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as error:
            errors.append(f"{prefix}: unreadable image: {error}")
            continue
        for class_name in class_names:
            class_boxes = boxes.get(class_name, [])
            if not isinstance(class_boxes, list):
                errors.append(f"{prefix}: {class_name} must contain a list")
                continue
            for box_index, box in enumerate(class_boxes):
                if not isinstance(box, list) or len(box) != 4 or not all(isinstance(value, (int, float)) for value in box):
                    errors.append(f"{prefix}: invalid box {class_name}[{box_index}]")
                    continue
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append(f"{prefix}: out-of-range box {class_name}[{box_index}] for {width}x{height}")
    return errors, len(samples)


def main():
    parser = shared_parser("Validate detection dataset")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = dataset_paths(args.data_dir)
    total = 0
    all_errors = []
    for split in ("train", "validation", "test"):
        errors, count = validate_split(paths[f"{split}_json"], paths[f"{split}_images"], config["class_names"])
        total += count
        all_errors.extend(errors)
        print(f"{split}: {count} samples, {len(errors)} errors")
    if all_errors:
        print("\n".join(all_errors))
        sys.exit(1)
    print(f"Dataset is valid: {total} samples")


if __name__ == "__main__":
    main()
