"""
Batch inference: recursively process all images in a folder with a text prompt
Usage example:
  python batch_inference.py --input_dir ./segmentation_input --output_dir segmentation_output --text "person" --model_dir ./onnx-models --tokenizer ./onnx-models/tokenizer.json --device cuda
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
from inference import Sam3ONNXInference, save_coords_txt, visualize_results


def list_images_recursive(folder: Path, exts=None):
    """Recursively find all images in folder and subfolders"""
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    return sorted(images)


def load_existing_json(out_path: Path):
    """Load existing LabelMe JSON if it exists"""
    out_path = Path(out_path)
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_labelme_json_local(image_path: Path, boxes, scores, prompt: str, out_path: Path):
    # boxes: numpy array [N,4] x1,y1,x2,y2
    image_path = Path(image_path)
    out_path = Path(out_path)

    img = Image.open(image_path)
    w, h = img.size
    
    # Load existing JSON if it exists
    existing_data = load_existing_json(out_path)
    
    if existing_data:
        # JSON exists, update or append shapes
        shapes = existing_data.get("shapes", [])
        existing_labels = set()
        
        # Update existing shapes with the same label
        updated = False
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [float(v) for v in box]
                new_shape = {
                    "label": prompt,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": f"score:{float(scores[i]):.4f}",
                    "shape_type": "rectangle",
                    "flags": {},
                }
                
                # Check if there's an existing shape with the same label
                found = False
                for j, existing_shape in enumerate(shapes):
                    if existing_shape.get("label") == prompt:
                        # Update the first shape with matching label
                        shapes[j] = new_shape
                        found = True
                        updated = True
                        break
                
                if not found:
                    # Append as new shape
                    shapes.append(new_shape)
                    updated = True
        
        # Keep shapes with different labels
        final_shapes = [s for s in shapes if s.get("label") != prompt]
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [float(v) for v in box]
                new_shape = {
                    "label": prompt,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": f"score:{float(scores[i]):.4f}",
                    "shape_type": "rectangle",
                    "flags": {},
                }
                final_shapes.append(new_shape)
    else:
        # No existing JSON, create new shapes
        final_shapes = []
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [float(v) for v in box]
                shape = {
                    "label": prompt,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": f"score:{float(scores[i]):.4f}",
                    "shape_type": "rectangle",
                    "flags": {},
                }
                final_shapes.append(shape)

    data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": final_shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Batch SAM3 ONNX inference and save LabelMe JSON")
    parser.add_argument("--input_dir", required=True, dest="input_dir", help="Main input image folder",)
    parser.add_argument("--output_dir", default="segmentation_output", help="Output directory for results")
    parser.add_argument("--text", required=True, help="Text prompt")
    parser.add_argument("--model_dir", default="onnx-models", help="ONNX models directory")
    parser.add_argument("--tokenizer", default=str(Path("onnx-models") / "tokenizer.json"), help="Path to tokenizer.json")
    parser.add_argument("--device", default="cuda", help="Device: 'cuda', 'npu' (CANN), or 'cpu'")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for detections")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    engine = Sam3ONNXInference(
        vision_encoder_path=str(model_dir / "vision-encoder.onnx"),
        text_encoder_path=str(model_dir / "text-encoder.onnx"),
        geometry_encoder_path=str(model_dir / "geometry-encoder.onnx"),
        decoder_path=str(model_dir / "decoder.onnx"),
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    images = list_images_recursive(input_dir)
    if not images:
        raise SystemExit(f"No images found in {input_dir} or its subfolders")

    print(f"Found {len(images)} images in {input_dir} and its subfolders")
    print(f"Processing with prompt: '{args.text}'")
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
            out_json = out_subdir / f"{p.stem}_result.json"
            save_labelme_json_local(p, boxes, scores, args.text, out_json)
            
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
