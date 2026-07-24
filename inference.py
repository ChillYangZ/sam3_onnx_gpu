"""
SAM3 ONNX Inference Script
Usage example:
  python inference.py --input_image ./segmentation_input/test.jpg --output_dir segmentation_output --text "person" --model_dir ./onnx-models --tokenizer ./onnx-models/tokenizer.json --device cuda
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

try:
    from tokenizers import Tokenizer
except Exception:
    Tokenizer = None

try:
    from transformers import PreTrainedTokenizerFast
except Exception:
    PreTrainedTokenizerFast = None

TARGET_SIZE = 1008


def parse_box_prompts(box_str: str) -> tuple[list, list]:
    """Parse box prompts string

    Format: "pos:x,y,w,h;neg:x,y,w,h;..." (xywh format)
    Returns: boxes [[x,y,w,h], ...], labels [1, 0, ...]
    """
    boxes, labels = [], []
    for part in box_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("pos:"):
            label, coords = 1, part[4:]
        elif part.startswith("neg:"):
            label, coords = 0, part[4:]
        else:
            label, coords = 1, part  # default positive
        x, y, w, h = [float(v) for v in coords.split(",")]
        boxes.append([x, y, w, h])
        labels.append(label)
    return boxes, labels


def xywh_to_cxcywh_normalized(boxes: list, img_w: int, img_h: int) -> np.ndarray:
    """Convert xywh (pixel) to cxcywh (normalized)"""
    result = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        result.append([cx, cy, nw, nh])
    return np.array(result, dtype=np.float32)


class Sam3ONNXInference:
    """SAM3 ONNX Inference Engine"""

    def __init__(
        self,
        vision_encoder_path: str,
        text_encoder_path: str,
        geometry_encoder_path: str,
        decoder_path: str,
        tokenizer_path: str,
        device: str = "cuda",
    ):
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device in ("npu", "cann"):
            providers = ["CANNExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        print("Loading ONNX models...")
        self.vision_encoder = ort.InferenceSession(
            vision_encoder_path, providers=providers
        )
        self.text_encoder = ort.InferenceSession(text_encoder_path, providers=providers)
        self.geometry_encoder = ort.InferenceSession(
            geometry_encoder_path, providers=providers
        )
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)
        # Load tokenizer: prefer `tokenizers.Tokenizer`, fallback to
        # `transformers.PreTrainedTokenizerFast` when parsing fails.
        self.is_transformers_tokenizer = False
        if Tokenizer is not None:
            try:
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
                self.tokenizer.enable_padding(length=32, pad_id=49407)
                self.tokenizer.enable_truncation(max_length=32)
            except Exception as e:
                print(f"Tokenizer.from_file failed: {e}; falling back to transformers tokenizer")
                self.tokenizer = None
        else:
            self.tokenizer = None

        if self.tokenizer is None:
            if PreTrainedTokenizerFast is None:
                raise RuntimeError("No usable tokenizer available: install 'tokenizers' or 'transformers'.")
            # Use transformers fast tokenizer wrapper
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            self.is_transformers_tokenizer = True
        print("  ✓ All models loaded")

    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess: resize to target size and normalize"""
        orig_size = image.shape[:2]  # (h, w)
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)
        resized = np.array(
            pil_image.resize((TARGET_SIZE, TARGET_SIZE), PILImage.BILINEAR)
        )
        normalized = resized.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
        tensor = normalized.transpose(2, 0, 1)[np.newaxis]  # NCHW
        return tensor, orig_size

    def encode_image(self, pixel_values: np.ndarray) -> dict:
        """Encode image using vision encoder"""
        outputs = self.vision_encoder.run(None, {"images": pixel_values})
        return {
            "fpn_feat_0": outputs[0],  # [B, 256, 288, 288]
            "fpn_feat_1": outputs[1],  # [B, 256, 144, 144]
            "fpn_feat_2": outputs[2],  # [B, 256, 72, 72]
            "fpn_pos_2": outputs[3],  # [B, 256, 72, 72]
        }

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode text prompt"""
        if self.is_transformers_tokenizer:
            encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=32)
            input_ids = np.array([encoded['input_ids']], dtype=np.int64)
            attention_mask = np.array([encoded['attention_mask']], dtype=np.int64)
        else:
            self.tokenizer.enable_padding(pad_id=49407, length=32)
            self.tokenizer.enable_truncation(max_length=32)
            encoded = self.tokenizer.encode(text)
            input_ids = np.array([encoded.ids], dtype=np.int64)
            attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        outputs = self.text_encoder.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return outputs[0], outputs[1]

    def encode_boxes(
        self,
        boxes: np.ndarray,
        labels: np.ndarray,
        fpn_feat: np.ndarray,
        fpn_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode box prompts"""
        outputs = self.geometry_encoder.run(
            None,
            {
                "input_boxes": boxes.astype(np.float32),
                "input_boxes_labels": labels.astype(np.int64),
                "fpn_feat_2": fpn_feat,
                "fpn_pos_2": fpn_pos,
            },
        )
        return outputs[0], outputs[1]

    def decode(
        self,
        vision_features: dict,
        prompt_features: np.ndarray,
        prompt_mask: np.ndarray,
    ) -> dict:
        """Decode features to generate masks"""
        outputs = self.decoder.run(
            None,
            {
                "fpn_feat_0": vision_features["fpn_feat_0"],
                "fpn_feat_1": vision_features["fpn_feat_1"],
                "fpn_feat_2": vision_features["fpn_feat_2"],
                "fpn_pos_2": vision_features["fpn_pos_2"],
                "prompt_features": prompt_features,
                "prompt_mask": prompt_mask,
            },
        )
        return {
            "pred_masks": outputs[0],
            "pred_boxes": outputs[1],
            "pred_logits": outputs[2],
            "presence_logits": outputs[3],
        }

    def predict(
        self,
        image: np.ndarray,
        text: Optional[str] = None,
        boxes: Optional[list] = None,
        box_labels: Optional[list] = None,
        conf_threshold: float = 0.3,
    ) -> dict:
        """Unified prediction with text and/or box prompts

        Args:
            image: RGB image [H, W, 3]
            text: Text prompt (optional)
            boxes: Box prompts [[x,y,w,h], ...] in xywh pixel format (optional)
            box_labels: Box labels [1, 0, ...] 1=pos, 0=neg (optional)
            conf_threshold: Confidence threshold
        """
        pixel_values, orig_size = self.preprocess_image(image)
        vision_features = self.encode_image(pixel_values)
        h, w = orig_size

        # Encode text
        if text:
            text_features, text_mask = self.encode_text(text)
        else:
            # No text: use padding tokens (length=32)
            pad_ids = np.full((1, 32), 49407, dtype=np.int64)
            pad_mask = np.zeros((1, 32), dtype=np.int64)
            pad_mask[0, 0] = 1  # at least one valid token
            outputs = self.text_encoder.run(
                None, {"input_ids": pad_ids, "attention_mask": pad_mask}
            )
            text_features, text_mask = outputs[0], outputs[1]

        # Encode boxes
        if boxes and len(boxes) > 0:
            boxes_cxcywh = xywh_to_cxcywh_normalized(boxes, w, h)
            boxes_array = boxes_cxcywh.reshape(1, -1, 4)
            if box_labels:
                labels_array = np.array(box_labels, dtype=np.int64).reshape(1, -1)
            else:
                labels_array = np.ones((1, len(boxes)), dtype=np.int64)
            geom_features, geom_mask = self.encode_boxes(
                boxes_array,
                labels_array,
                vision_features["fpn_feat_2"],
                vision_features["fpn_pos_2"],
            )
            # Concatenate text and geometry features
            prompt_features = np.concatenate([text_features, geom_features], axis=1)
            prompt_mask = np.concatenate([text_mask, geom_mask], axis=1)
        else:
            # No boxes: use text features only
            prompt_features = text_features
            prompt_mask = text_mask

        outputs = self.decode(vision_features, prompt_features, prompt_mask)
        return self._postprocess(outputs, orig_size, conf_threshold, boxes)

    def _postprocess(
        self,
        outputs: dict,
        orig_size: tuple[int, int],
        conf_threshold: float,
        input_boxes: Optional[list] = None,
    ) -> dict:
        """Post-process model outputs"""
        pred_masks = outputs["pred_masks"][0]
        pred_boxes = outputs["pred_boxes"][0]
        pred_logits = outputs["pred_logits"][0]
        presence_logits = outputs["presence_logits"][0, 0]

        presence_score = 1 / (1 + np.exp(-presence_logits))
        scores = (1 / (1 + np.exp(-pred_logits))) * presence_score
        keep = scores > conf_threshold

        h, w = orig_size
        # Resize masks: 288x288 -> original size
        masks = []
        for m in pred_masks[keep]:
            mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            masks.append(mask_resized > 0)
        
        # If no detections, return None for boxes and scores
        if not np.any(keep):
            return {
                "masks": masks,
                "boxes": None,
                "scores": None,
                "orig_size": orig_size,
                "input_boxes": input_boxes,
            }
            
        # Scale boxes from normalized [0,1] to pixel coordinates
        boxes = pred_boxes[keep].copy()
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        boxes = np.clip(boxes, 0, [[w, h, w, h]])

        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores[keep],
            "orig_size": orig_size,
            "input_boxes": input_boxes,
        }


def visualize_results(image: np.ndarray, results: dict, output_path: str, image_path: str = None, alpha: float = 0.35):
    """Visualize detection results with mask overlay and contours

    Args:
        image: Input image array
        results: Inference results dict
        output_path: Output directory path
        image_path: Input image file path (used to generate output filename)
        alpha: Mask overlay transparency
    """
    vis = image.copy()
    colors = [
        (30, 144, 255),  # Dodger Blue
        (255, 144, 30),  # Orange
        (144, 255, 30),  # Green-Yellow
        (255, 30, 144),  # Pink
        (30, 255, 144),  # Spring Green
    ]

    masks = results["masks"]
    boxes = results.get("boxes", [])
    scores = results.get("scores", [])

    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        mask_bool = mask > 0

        # Apply mask overlay (lighter)
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        # Draw mask contours
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

        # Draw box and score if available
        if boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if scores is not None and i < len(scores):
                cv2.putText(
                    vis, f"{scores[i]:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Generate output filename: input_name_output.ext
    if image_path:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(os.path.basename(image_path))[1] or ".png"
        output_filename = f"{base_name}_output{ext}"
    else:
        output_filename = "output.png"

    full_output_path = os.path.join(output_path, output_filename)

    cv2.imwrite(full_output_path, vis)
    print(f"  ✓ Saved: {full_output_path}")


def save_coords_txt(image_path: str, boxes, scores, prompt: str, out_path: str):
    """Save coordinates to txt file"""
    with open(out_path, "w") as f:
        f.write(f"# 图像: {os.path.basename(image_path)}\n")
        f.write(f"# 文本提示: {prompt}\n")
        if boxes is None or len(boxes) == 0:
            f.write("# 检测到 0 个对象\n")
        else:
            f.write(f"# 检测到 {len(boxes)} 个对象\n")
            f.write("# 格式: 索引 x1 y1 x2 y2\n\n")
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                score_str = f"  score: {scores[idx]:.4f}" if scores is not None and idx < len(scores) else ""
                f.write(f"{idx} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}{score_str}\n")
    print(f"  ✓ Saved: {out_path}")


def save_labelme_json(image_path: str, boxes, scores, prompt: str, out_path: str):
    """Save LabelMe format JSON file"""
    img = Image.open(image_path)
    image_width, image_height = img.size
    img.close()
    
    shapes = []
    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            shape = {
                "label": prompt if prompt else "object",
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": f"score: {scores[idx]:.4f}" if scores is not None and idx < len(scores) else "",
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)
    
    json_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 ONNX Inference")
    parser.add_argument("--input_image", type=str, required=True, help="Input image path")
    parser.add_argument("--output_dir", type=str, default="segmentation_output", help="Output path")
    parser.add_argument("--text", type=str, help="Text prompt")
    parser.add_argument(
        "--boxes", type=str, help="Box prompts: pos:x,y,w,h;neg:x,y,w,h (xywh format)"
    )
    parser.add_argument(
        "--model_dir", type=str, default="onnx-models", help="ONNX models directory"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to tokenizer.json"
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda', 'npu' (CANN), or 'cpu'")
    args = parser.parse_args()

    if not args.text and not args.boxes:
        parser.error("Please specify --text or --boxes")

    # Load model
    model_dir = Path(args.model_dir)
    engine = Sam3ONNXInference(
        vision_encoder_path=str(model_dir / "vision-encoder.onnx"),
        text_encoder_path=str(model_dir / "text-encoder.onnx"),
        geometry_encoder_path=str(model_dir / "geometry-encoder.onnx"),
        decoder_path=str(model_dir / "decoder.onnx"),
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    # Load image
    image_bgr = cv2.imread(args.input_image)
    if image_bgr is None:
        raise ValueError(f"Cannot load image: {args.input_image}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"\nProcessing: {args.input_image} ({image.shape[1]}x{image.shape[0]})")

    # Parse prompts
    boxes, box_labels = None, None
    if args.boxes:
        boxes, box_labels = parse_box_prompts(args.boxes)
        print(f"  Box prompts: {len(boxes)} boxes, labels={box_labels}")
    if args.text:
        print(f"  Text prompt: '{args.text}'")

    # Run inference
    results = engine.predict(
        image,
        text=args.text,
        boxes=boxes,
        box_labels=box_labels,
        conf_threshold=args.conf,
    )

    print(f"  Found {len(results['masks'])} objects")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save visualization image to output_dir
    visualize_results(image_bgr, results, args.output_dir, args.input_image)
    
    # Save results to txt and json
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]
    boxes = results.get("boxes")
    scores = results.get("scores")
    
    out_txt = os.path.join(args.output_dir, f"{base_name}_coords.txt")
    save_coords_txt(args.input_image, boxes, scores, args.text, out_txt)
    
    out_json = os.path.join(args.output_dir, f"{base_name}_result.json")
    save_labelme_json(args.input_image, boxes, scores, args.text, out_json)


if __name__ == "__main__":
    main()