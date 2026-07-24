"""
Batch inference: process all images in a folder with a text prompt
Usage example:
  python FP16batch_inference.py --input_dir ./segmentation_input --output_dir segmentation_output --text "person" --model_dir ./onnx-models
   --tokenizer ./onnx-models_export/tokenizer.json --device cuda
"""
import argparse
import time
import numpy as np
from pathlib import Path
from PIL import Image
from inference import Sam3ONNXInference, save_coords_txt, save_labelme_json, visualize_results


def list_images(folder: Path, exts=None):
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in exts and p.is_file()]


def main():
    parser = argparse.ArgumentParser(description="Batch SAM3 ONNX inference and save LabelMe JSON")
    parser.add_argument(
        "--input_dirs",
        "--input_dir",
        required=True,
        nargs="+",
        dest="input_dirs",
        help="One or more input image folders (space-separated)",
    )
    parser.add_argument("--text", required=True, help="Text prompt")
    parser.add_argument("--model_dir", default="onnx-models", help="ONNX models directory")
    parser.add_argument("--tokenizer", default=str(Path("onnx-models") / "tokenizer.json"), help="Path to tokenizer.json")
    parser.add_argument("--device", default="cuda", help="Device: 'cuda', 'npu' (CANN), or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--output_dir", default="batch_output", help="Output directory for results")
    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.input_dirs]
    valid_dirs = [d for d in input_dirs if d.is_dir()]
    if not valid_dirs:
        raise SystemExit(f"No valid input dirs found: {input_dirs}")

    model_dir = Path(args.model_dir)
    engine = Sam3ONNXInference(
        vision_encoder_path=str(model_dir / "vision-encoder-fp16.onnx"),
        text_encoder_path=str(model_dir / "text-encoder-fp16.onnx"),
        geometry_encoder_path=str(model_dir / "geometry-encoder-fp16.onnx"),
        decoder_path=str(model_dir / "decoder-fp16.onnx"),
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each valid input directory in order
    for input_dir in valid_dirs:
        images = list_images(input_dir)
        if not images:
            print(f"No images found in {input_dir}")
            continue

        print(f"Processing {len(images)} images in {input_dir} with prompt: '{args.text}'")
        print(f"Output directory: {output_dir}")
        
        for p in images:
            try:
                # Create relative output path to maintain folder structure
                rel_path = p.relative_to(input_dir)
                out_subdir = output_dir / rel_path.parent
                out_subdir.mkdir(parents=True, exist_ok=True)
                
                img = Image.open(p).convert("RGB")
                img_np = np.array(img)

                start = time.time()
                results = engine.predict(img_np, text=args.text, conf_threshold=args.conf)
                elapsed = time.time() - start

                boxes = results.get("boxes")
                scores = results.get("scores")
                
                # Save JSON to output_dir
                out_json = str(out_subdir / f"{p.stem}_result.json")
                save_labelme_json(str(p), boxes, scores, args.text, out_json)
                
                # Save coords txt to output_dir
                out_txt = str(out_subdir / f"{p.stem}_coords.txt")
                save_coords_txt(str(p), boxes, scores, args.text, out_txt)
                
                # Save visualization image to output_dir
                out_img_dir = str(out_subdir)
                visualize_results(img_np, results, out_img_dir, str(p))

                num = 0
                try:
                    if boxes is not None:
                        num = len(boxes)
                except Exception:
                    num = 0

                print(f"{rel_path}: inference_time={elapsed:.3f}s, detections={num}, saved to {out_subdir}")
            except Exception as e:
                print(f"Failed {p}: {e}")


if __name__ == "__main__":
    main()
