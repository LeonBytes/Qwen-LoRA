import json
import math
import os
import re

import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


def parse_predictions(response, class_names):
    # 模型偶尔会在 JSON 外包裹 Markdown 代码块，这里统一清理后再解析。
    cleaned = re.sub(r"```(?:json)?", "", response).replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        return {name: [] for name in class_names}
    try:
        parsed = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return {name: [] for name in class_names}
    result = {}
    for name in class_names:
        boxes = parsed.get(name, [])
        result[name] = [
            [float(value) for value in box]
            for box in boxes
            if isinstance(box, list) and len(box) == 4
        ]
    return result


def calculate_iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(ground_truth, predictions, threshold):
    # 使用二分图增广匹配，保证每个预测框和真实框最多匹配一次。
    candidates = [
        [index for index, target in enumerate(ground_truth) if calculate_iou(prediction, target) >= threshold]
        for prediction in predictions
    ]
    assignments = {}

    def augment(prediction_index, visited):
        for target_index in candidates[prediction_index]:
            if target_index in visited:
                continue
            visited.add(target_index)
            if target_index not in assignments or augment(assignments[target_index], visited):
                assignments[target_index] = prediction_index
                return True
        return False

    true_positive = sum(augment(index, set()) for index in range(len(predictions)))
    return true_positive, len(predictions) - true_positive, len(ground_truth) - true_positive


def build_report(counters, threshold, sample_count):
    per_class = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for class_name, values in counters.items():
        precision = values["tp"] / max(1, values["tp"] + values["fp"])
        recall = values["tp"] / max(1, values["tp"] + values["fn"])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[class_name] = {**values, "precision": precision, "recall": recall, "f1": f1}
        for key in totals:
            totals[key] += values[key]
    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    return {
        "sample_count": sample_count,
        "iou_threshold": threshold,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(per_class),
        "micro": {**totals, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-12, precision + recall)},
        "per_class": per_class,
    }


def evaluate_dataset(model, processor, annotation_path, image_dir, prompt, class_names, threshold, max_new_tokens):
    with open(annotation_path, "r", encoding="utf-8") as file:
        samples = json.load(file)
    counters = {name: {"tp": 0, "fp": 0, "fn": 0} for name in class_names}
    records = []
    model.eval()
    for sample in tqdm(samples, desc="Evaluating"):
        image_path = os.path.join(image_dir, sample["image"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        predictions = parse_predictions(response, class_names)
        # gpt.value 是外层标注 JSON 中保存的一段 JSON 字符串，需要再次解析。
        ground_truth = json.loads(sample["conversations"][1]["value"])
        for class_name in class_names:
            tp, fp, fn = match_boxes(ground_truth.get(class_name, []), predictions[class_name], threshold)
            counters[class_name]["tp"] += tp
            counters[class_name]["fp"] += fp
            counters[class_name]["fn"] += fn
        records.append({"image": sample["image"], "ground_truth": ground_truth, "predictions": predictions, "raw_response": response})
    return build_report(counters, threshold, len(samples)), records


def draw_boxes(image, boxes_by_class, class_names, colors, title):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, 28), fill="white")
    draw.text((8, 8), title, fill="black", font=font)
    for class_name in class_names:
        for box in boxes_by_class.get(class_name, []):
            if not isinstance(box, list) or len(box) != 4:
                continue
            values = [float(value) for value in box]
            if not all(math.isfinite(value) for value in values):
                continue
            x1, x2 = sorted((max(0, min(values[0], canvas.width)), max(0, min(values[2], canvas.width))))
            y1, y2 = sorted((max(0, min(values[1], canvas.height)), max(0, min(values[3], canvas.height))))
            if x2 <= x1 or y2 <= y1:
                continue
            color = colors[class_name]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            label_y = max(30, int(y1) - 14)
            draw.rectangle((int(x1), label_y, int(x1) + 8 * len(class_name), label_y + 13), fill="white")
            draw.text((int(x1), label_y), class_name, fill=color, font=font)
    return canvas


def save_visualizations(records, image_dir, output_dir, class_names):
    palette = ("red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow")
    colors = {name: palette[index % len(palette)] for index, name in enumerate(class_names)}
    os.makedirs(output_dir, exist_ok=True)
    for index, record in enumerate(tqdm(records, desc="Visualizing")):
        with Image.open(os.path.join(image_dir, record["image"])) as source:
            image = source.convert("RGB")
        target = draw_boxes(image, record["ground_truth"], class_names, colors, "Ground truth")
        prediction = draw_boxes(image, record["predictions"], class_names, colors, "Prediction")
        comparison = Image.new("RGB", (target.width * 2, target.height), "white")
        comparison.paste(target, (0, 0))
        comparison.paste(prediction, (target.width, 0))
        stem = os.path.splitext(os.path.basename(record["image"]))[0]
        comparison.save(os.path.join(output_dir, f"{index:04d}_{stem}.png"))
