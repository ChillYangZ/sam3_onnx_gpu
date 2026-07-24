"""
Parallel batch inference: recursively process images with a text prompt.
Usage example:
  python parallel_batch_inference.py --input_dir ./segmentation_input --output_dir segmentation_output --text "person" --model_dir ./onnx-models --tokenizer ./onnx-models/tokenizer.json --device cuda --num_processes 4
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from batch_inference import save_labelme_json_local
from inference import Sam3ONNXInference, save_coords_txt, visualize_results


def list_images_recursive(folder: Path, exts=None):
    """Recursively find all images in folder and subfolders."""
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    return sorted(images)


def split_list(items: list, num_splits: int):
    """Split a list into num_splits chunks as evenly as possible."""
    if num_splits <= 1 or len(items) <= 1:
        return [items]

    num_splits = min(num_splits, len(items))
    base_size = len(items) // num_splits
    remainder = len(items) % num_splits

    chunks = []
    start_idx = 0
    for i in range(num_splits):
        end_idx = start_idx + base_size + (1 if i < remainder else 0)
        chunks.append(items[start_idx:end_idx])
        start_idx = end_idx
    return chunks


def collect_image_tasks(input_dirs: Iterable[Path]):
    tasks = []
    for input_dir in input_dirs:
        images = list_images_recursive(input_dir)
        if not images:
            print(f"No images found in {input_dir}")
            continue
        print(f"Found {len(images)} images in {input_dir} and its subfolders")
        for image_path in images:
            tasks.append((str(image_path), str(input_dir)))
    return tasks


def process_image_chunk(
    image_tasks: list[tuple[str, str]],
    output_dir: str,
    text: str,
    model_dir: str,
    tokenizer: str,
    device: str,
    conf: float,
):
    model_path = Path(model_dir)
    engine = Sam3ONNXInference(
        vision_encoder_path=str(model_path / "vision-encoder.onnx"),
        text_encoder_path=str(model_path / "text-encoder.onnx"),
        geometry_encoder_path=str(model_path / "geometry-encoder.onnx"),
        decoder_path=str(model_path / "decoder.onnx"),
        tokenizer_path=tokenizer,
        device=device,
    )

    output_path = Path(output_dir)
    messages = []

    for image_path_str, input_dir_str in image_tasks:
        p = Path(image_path_str)
        input_dir = Path(input_dir_str)
        try:
            rel_path = p.relative_to(input_dir)
            out_subdir = output_path / rel_path.parent
            out_subdir.mkdir(parents=True, exist_ok=True)

            img = Image.open(p).convert("RGB")
            img_np = np.array(img)

            start = time.time()
            results = engine.predict(img_np, text=text, conf_threshold=conf)
            elapsed = time.time() - start

            boxes = results.get("boxes")
            scores = results.get("scores")

            out_json = out_subdir / f"{p.stem}_result.json"
            save_labelme_json_local(p, boxes, scores, text, out_json)

            out_txt = str(out_subdir / f"{p.stem}_coords.txt")
            save_coords_txt(str(p), boxes, scores, text, out_txt)

            out_img_dir = str(out_subdir)
            visualize_results(img_np, results, out_img_dir, str(p))

            num = 0
            try:
                if boxes is not None:
                    num = len(boxes)
            except Exception:
                num = 0

            messages.append(f"{rel_path}: inference_time={elapsed:.3f}s, detections={num}, saved to {out_subdir}")
        except Exception as e:
            messages.append(f"Failed {p}: {e}")

    return messages


def main():
    parser = argparse.ArgumentParser(description="Parallel batch SAM3 ONNX inference and save LabelMe JSON")
    parser.add_argument(
        "--input_dirs",
        "--input_dir",
        "--input-dirs",
        "--input-dir",
        required=True,
        nargs="+",
        dest="input_dirs",
        help="One or more input image folders (space-separated)",
    )
    parser.add_argument("--output_dir", "--output-dir", default="segmentation_output", help="Output directory for results")
    parser.add_argument("--text", required=True, help="Text prompt")
    parser.add_argument("--model_dir", "--model-dir", default="onnx-models", help="ONNX models directory")
    parser.add_argument("--tokenizer", default=str(Path("onnx-models") / "tokenizer.json"), help="Path to tokenizer.json")
    parser.add_argument("--device", default="cuda", help="Device: 'cuda', 'npu' (CANN), or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--num_processes", "--num-processes", type=int, default=4, help="Number of parallel processes to run")
    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    valid_dirs = [d for d in input_dirs if d.is_dir()]
    if not valid_dirs:
        raise SystemExit(f"No valid input dirs found: {input_dirs}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_tasks = collect_image_tasks(valid_dirs)
    if not image_tasks:
        raise SystemExit(f"No images found in any of the input directories: {valid_dirs}")

    num_processes = max(1, args.num_processes)
    image_chunks = split_list(image_tasks, num_processes)

    print(f"Found {len(image_tasks)} total images across {len(valid_dirs)} directories")
    print(f"Processing with prompt: '{args.text}'")
    print(f"Output directory: {output_dir}")
    print(f"Using {len(image_chunks)} parallel processes")

    with ProcessPoolExecutor(max_workers=len(image_chunks)) as executor:
        futures = [
            executor.submit(
                process_image_chunk,
                chunk,
                str(output_dir),
                args.text,
                args.model_dir,
                args.tokenizer,
                args.device,
                args.conf,
            )
            for chunk in image_chunks
            if chunk
        ]

        for future in as_completed(futures):
            for message in future.result():
                print(message)

    print("All processes completed!")


if __name__ == "__main__":
    main()
