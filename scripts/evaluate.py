import json
import os

from common import dataset_paths, load_config, load_inference_model, require_paths, shared_parser
from metrics import evaluate_dataset, save_visualizations


def main():
    parser = shared_parser("Evaluate a Qwen2.5-VL LoRA adapter")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--skip-visualizations", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = dataset_paths(args.data_dir)
    require_paths(paths, ("test_json", "test_images"))
    adapter_path = args.adapter_path or os.path.join(args.output_dir, "best_model")
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(adapter_path)
    model, processor = load_inference_model(config, args.model_path, adapter_path)
    report, records = evaluate_dataset(
        model,
        processor,
        paths["test_json"],
        paths["test_images"],
        config["prompt"],
        config["class_names"],
        config.get("iou_threshold", 0.5),
        config.get("max_new_tokens", 512),
    )
    result_dir = os.path.join(args.output_dir, "test_evaluation")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "evaluation_report.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    with open(os.path.join(result_dir, "predictions.json"), "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
    if not args.skip_visualizations:
        save_visualizations(records, paths["test_images"], os.path.join(result_dir, "visualizations"), config["class_names"])
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
