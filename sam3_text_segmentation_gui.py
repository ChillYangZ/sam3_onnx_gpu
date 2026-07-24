"""
SAM 3 文本提示分割 GUI 应用

本应用提供了一个基于PyQt的图形界面，用于使用SAM 3模型进行文本提示的图像分割。
用户可以输入文本提示，选择单张图片或整个文件夹，然后点击开始进行分割。
分割结果的边界框坐标将保存为txt文件。

确保您已安装必要的依赖：
pip install torch pillow matplotlib opencv-python pyqt5 onnxruntime-gpu
"""
import json
import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
import onnxruntime as ort
from tokenizers import Tokenizer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QRadioButton,
    QButtonGroup,
    QMessageBox,
    QSplitter,
    QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# SAM3 ONNX相关常量
TARGET_SIZE = 1008


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
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self.vision_encoder = ort.InferenceSession(vision_encoder_path, providers=providers)
        self.text_encoder = ort.InferenceSession(text_encoder_path, providers=providers)
        self.geometry_encoder = ort.InferenceSession(geometry_encoder_path, providers=providers)
        self.decoder = ort.InferenceSession(decoder_path, providers=providers)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(length=32, pad_id=49407)
        self.tokenizer.enable_truncation(max_length=32)

    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Preprocess: resize to target size and normalize"""
        orig_size = image.shape[:2]  # (h, w)
        pil_image = Image.fromarray(image)
        resized = np.array(
            pil_image.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
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
        text: str = None,
        boxes: list = None,
        box_labels: list = None,
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
        input_boxes: list = None,
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

class SegmentationThread(QThread):
    """
    分割线程类，用于在后台执行分割任务
    """
    progress_updated = pyqtSignal(int, str)  # 当前进度，状态信息
    finished = pyqtSignal(bool, str)  # 是否成功，结果信息
    preview_available = pyqtSignal(object, object, object, str)  # 预览信号：图像、掩码、边界框、图像路径
    
    def __init__(self, onnx_engine, text_prompt, paths, output_dir, confidence_threshold=0.25):
        super().__init__()
        self.onnx_engine = onnx_engine
        self.text_prompt = text_prompt
        self.paths = paths
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.running = True
    
    def run(self):
        """
        执行分割任务
        """
        try:
            total = len(self.paths)
            for i, image_path in enumerate(self.paths):
                if not self.running:
                    break
                
                self.progress_updated.emit(int((i+1)/total*100), f"处理中: {os.path.basename(image_path)}")
                
                # 加载和预处理图像
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)
                
                # 开始推理计时
                start_time = time.time()
                
                # 进行分割
                output = self.onnx_engine.predict(
                    image=image_np,
                    text=self.text_prompt,
                    conf_threshold=self.confidence_threshold
                )
                
                # 获取结果
                masks = np.array(output["masks"])
                boxes = output["boxes"]
                scores = output["scores"]
                
                # 计算推理耗时
                end_time = time.time()
                inference_time = end_time - start_time
                
                # 更新进度并显示耗时
                self.progress_updated.emit(int((i+1)/total*100), f"处理完成: {os.path.basename(image_path)} (推理耗时: {inference_time:.2f}秒)")
                
                # 发送预览信号
                self.preview_available.emit(image, masks, boxes, image_path)
                
                # 保存分割坐标
                self.save_segmentation_coords(image_path, boxes, scores)
            
            if self.running:
                self.finished.emit(True, f"分割完成！共处理 {len(self.paths)} 张图片")
            else:
                self.finished.emit(False, "分割已取消")
                
        except Exception as e:
            self.finished.emit(False, f"分割出错: {str(e)}")
    
    def save_segmentation_coords(self, image_path, boxes, scores):
        """
        保存分割坐标到txt和json文件
        
        Args:
            image_path: 图像文件路径
            boxes: 边界框列表，格式: [num_masks, 4]，格式: [x1, y1, x2, y2]
            scores: 置信度分数列表
        """
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        coords_file = os.path.join(self.output_dir, f"{base_name}.txt")
        json_file = os.path.join(self.output_dir, f"{base_name}.json")
        
        # 将boxes和scores转换为numpy数组
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes
        
        if isinstance(scores, torch.Tensor):
            scores_np = scores.cpu().numpy()
        else:
            scores_np = scores
        
        # 写入txt文件
        with open(coords_file, "w") as f:
            f.write(f"# 图像: {os.path.basename(image_path)}\n")
            f.write(f"# 文本提示: {self.text_prompt}\n")
            if boxes_np is None or len(boxes_np) == 0:
                f.write("# 检测到 0 个对象\n")
                f.write("# 格式: 索引 x1 y1 x2 y2\n\n")
            else:
                f.write(f"# 检测到 {len(boxes_np)} 个对象\n")
                f.write("# 格式: 索引 x1 y1 x2 y2\n\n")
                
                for idx, box in enumerate(boxes_np):
                    x1, y1, x2, y2 = box
                    f.write(f"{idx} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
        
        # 写入json文件（LabelMe格式）
        # 打开图像获取尺寸
        img = Image.open(image_path)
        image_width, image_height = img.size
        img.close()
        
        # 准备shapes数据
        shapes = []
        if boxes_np is not None and len(boxes_np) > 0:
            for idx, box in enumerate(boxes_np):
                x1, y1, x2, y2 = box
                shape = {
                    "label": self.text_prompt,
                    "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                }
                shapes.append(shape)
        
        # 准备JSON数据
        json_data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }
        
        # 写入JSON文件
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"保存完成: {json_file}")
                
    def stop(self):
        """
        停止分割任务
        """
        self.running = False

class SAM3SegmentationGUI(QMainWindow):
    """
    SAM3分割GUI主窗口
    """
    def __init__(self):
        super().__init__()
        self.onnx_engine = None
        self.segmentation_thread = None
        self.init_ui()
        self.init_model()
    
    def init_ui(self):
        """
        初始化UI
        """
        self.setWindowTitle("SAM3 文本提示图像分割")
        self.setGeometry(100, 100, 1000, 600)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主垂直布局
        main_layout = QVBoxLayout(central_widget)
        
        # 文本提示输入
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("文本提示:"))
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("例如: human, car, cat")
        prompt_layout.addWidget(self.prompt_input, 1)
        main_layout.addLayout(prompt_layout)
        
        # 选择模式
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("选择模式:"))
        
        self.radio_single = QRadioButton("单张图片")
        self.radio_single.setChecked(True)
        self.radio_folder = QRadioButton("文件夹")
        
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_single)
        self.mode_group.addButton(self.radio_folder)
        
        mode_layout.addWidget(self.radio_single)
        mode_layout.addWidget(self.radio_folder)
        mode_layout.addStretch()
        main_layout.addLayout(mode_layout)
        
        # 路径选择
        path_layout = QHBoxLayout()
        self.path_button = QPushButton("选择路径")
        self.path_button.clicked.connect(self.select_path)
        path_layout.addWidget(self.path_button)
        
        self.path_label = QLabel("未选择路径")
        self.path_label.setWordWrap(True)
        path_layout.addWidget(self.path_label, 1)
        main_layout.addLayout(path_layout)
        
        # 模型目录选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型目录:"))
        self.model_button = QPushButton("选择目录")
        self.model_button.clicked.connect(self.select_model_dir)
        model_layout.addWidget(self.model_button)
        
        self.model_label = QLabel(os.path.join(os.getcwd(), "onnx-models"))
        self.model_label.setWordWrap(True)
        model_layout.addWidget(self.model_label, 1)
        main_layout.addLayout(model_layout)
        
        # 输出目录选择
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出目录:"))
        self.output_button = QPushButton("选择目录")
        self.output_button.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_button)
        
        self.output_label = QLabel(os.path.join(os.getcwd(), "segmentation_output"))
        self.output_label.setWordWrap(True)
        output_layout.addWidget(self.output_label, 1)
        main_layout.addLayout(output_layout)
        
        # 置信度阈值控制
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("置信度阈值:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(25)  # 默认25%置信度
        self.confidence_slider.setTickInterval(5)
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setToolTip("调整检测对象的置信度阈值，值越高检测越严格")
        confidence_layout.addWidget(self.confidence_slider, 1)
        
        self.confidence_label = QLabel("0.25")
        confidence_layout.addWidget(self.confidence_label)
        # 连接滑块信号
        self.confidence_slider.valueChanged.connect(lambda value: self.confidence_label.setText(f"{value/100:.2f}"))
        main_layout.addLayout(confidence_layout)
        
        # 开始/取消按钮
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始分割")
        self.start_button.clicked.connect(self.start_segmentation)
        button_layout.addWidget(self.start_button)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel_segmentation)
        self.cancel_button.setEnabled(False)
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # 预览区域和状态信息
        preview_status_layout = QHBoxLayout()
        
        # 预览区域
        preview_layout = QVBoxLayout()
        preview_layout.addWidget(QLabel("分割结果预览:"))
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #cccccc; background-color: #f0f0f0;")
        self.preview_label.setMinimumHeight(400)
        preview_layout.addWidget(self.preview_label)
        
        # 状态信息
        status_layout = QVBoxLayout()
        status_layout.addWidget(QLabel("状态信息:"))
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(400)
        status_layout.addWidget(self.status_text)
        
        # 分割预览和状态区域 - 增大预览区域比例
        preview_status_layout.addLayout(preview_layout, 3)
        preview_status_layout.addLayout(status_layout, 1)
        main_layout.addLayout(preview_status_layout)
        
        # 初始化模型目录
        self.model_dir = os.path.join(os.getcwd(), "onnx-models")
        
        # 初始化输出目录
        self.output_dir = os.path.join(os.getcwd(), "segmentation_output")
        
        # 初始化路径列表
        self.image_paths = []
    
    def init_model(self):
        """
        初始化SAM3 ONNX模型
        """
        self.status_text.append("正在加载SAM3 ONNX模型...")
        QApplication.processEvents()
        
        try:
            # 自动选择设备，优先CUDA，然后CPU
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            self.status_text.append(f"使用设备: {device}")
            
            # 构建ONNX推理引擎
            model_dir = self.model_dir
            self.status_text.append(f"加载模型目录: {model_dir}")
            
            # 检查模型文件是否存在
            model_files = [
                os.path.join(model_dir, "vision-encoder.onnx"),
                os.path.join(model_dir, "text-encoder.onnx"),
                os.path.join(model_dir, "geometry-encoder.onnx"),
                os.path.join(model_dir, "decoder.onnx"),
                os.path.join(model_dir, "tokenizer.json"),
            ]
            
            missing_files = [f for f in model_files if not os.path.exists(f)]
            if missing_files:
                self.status_text.append(f"缺少模型文件: {missing_files}")
                self.status_text.append("请先下载ONNX模型到 onnx-models/ 目录")
                self.status_text.append("下载地址: https://github.com/jamjamjon/assets/releases/tag/sam3")
                return
            
            self.onnx_engine = Sam3ONNXInference(
                vision_encoder_path=os.path.join(model_dir, "vision-encoder.onnx"),
                text_encoder_path=os.path.join(model_dir, "text-encoder.onnx"),
                geometry_encoder_path=os.path.join(model_dir, "geometry-encoder.onnx"),
                decoder_path=os.path.join(model_dir, "decoder.onnx"),
                tokenizer_path=os.path.join(model_dir, "tokenizer.json"),
                device=device,
            )
            
            self.status_text.append("模型加载完成！")
        except Exception as e:
            self.status_text.append(f"模型加载失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
    
    def select_path(self):
        """
        选择图片或文件夹路径
        """
        if self.radio_single.isChecked():
            # 选择单张图片
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
            if file_path:
                self.image_paths = [file_path]
                self.path_label.setText(file_path)
        else:
            # 选择文件夹
            folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
            if folder_path:
                # 获取文件夹中的所有图片
                self.image_paths = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                    self.image_paths.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                                            if os.path.isfile(os.path.join(folder_path, f)) 
                                            and f.lower().endswith(tuple(ext[2:]))])
                self.path_label.setText(f"{folder_path} (共 {len(self.image_paths)} 张图片)")
    
    def select_model_dir(self):
        """
        选择模型目录
        """
        model_dir = QFileDialog.getExistingDirectory(self, "选择ONNX模型目录")
        if model_dir:
            self.model_dir = model_dir
            self.model_label.setText(model_dir)
            # 重新加载模型
            self.init_model()
    
    def select_output_dir(self):
        """
        选择输出目录
        """
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if output_dir:
            self.output_dir = output_dir
            self.output_label.setText(output_dir)
    
    def start_segmentation(self):
        """
        开始分割
        """
        # 检查文本提示
        text_prompt = self.prompt_input.text().strip()
        if not text_prompt:
            QMessageBox.warning(self, "警告", "请输入文本提示")
            return
        
        # 检查是否选择了路径
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请选择图片或文件夹")
            return
        
        # 检查模型是否加载成功
        if not self.onnx_engine:
            QMessageBox.warning(self, "警告", "模型未加载成功")
            return
        
        # 禁用控件
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.path_button.setEnabled(False)
        self.output_button.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.radio_single.setEnabled(False)
        self.radio_folder.setEnabled(False)
        
        # 清空状态和进度
        self.status_text.append("\n开始分割...")
        self.progress_bar.setValue(0)
        
        # 获取置信度阈值
        confidence_threshold = self.confidence_slider.value() / 100.0
        
        # 创建分割线程
        self.segmentation_thread = SegmentationThread(
            self.onnx_engine,
            text_prompt,
            self.image_paths,
            self.output_dir,
            confidence_threshold
        )
        self.segmentation_thread.progress_updated.connect(self.update_progress)
        self.segmentation_thread.finished.connect(self.segmentation_finished)
        self.segmentation_thread.preview_available.connect(self.update_preview)
        self.segmentation_thread.start()
    
    def cancel_segmentation(self):
        """
        取消分割
        """
        if self.segmentation_thread and self.segmentation_thread.isRunning():
            self.segmentation_thread.stop()
            self.status_text.append("正在取消分割...")
    
    def update_progress(self, progress, message):
        """
        更新进度
        """
        self.progress_bar.setValue(progress)
        self.status_text.append(message)
        QApplication.processEvents()
    
    def update_preview(self, image, masks, boxes, image_path):
        """
        更新预览图像
        
        Args:
            image: PIL图像对象
            masks: 掩码列表
            boxes: 边界框列表
            image_path: 图像路径
        """
        # 将PIL图像转换为numpy数组
        image_np = np.array(image)
        
        # 创建预览图像副本
        preview_image = image_np.copy()
        
        # 处理空检测结果的情况
        if masks is None or len(masks) == 0 or boxes is None or len(boxes) == 0:
            self.status_text.append(f"{os.path.basename(image_path)}: 未检测到对象")
            self.display_image(preview_image)
            return
        
        # 将CUDA张量转换为numpy数组（如果需要）
        if isinstance(masks, torch.Tensor):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = masks
        
        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes
        
        # 选择置信度最高的前5个掩码进行显示
        num_to_show = min(5, len(masks_np))
        
        # 绘制掩码和边界框
        for i in range(num_to_show):
            mask = masks_np[i]
            box = boxes_np[i]
            
            # 移除可能存在的批量维度
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            
            # 为掩码生成随机颜色
            color = [np.random.randint(0, 255) for _ in range(3)]
            
            # 将掩码应用到预览图像
            mask = mask.astype(bool)
            preview_image[mask] = preview_image[mask] * 0.6 + np.array(color) * 0.4
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(preview_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制索引
            cv2.putText(preview_image, f"{i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 显示预览图像
        self.display_image(preview_image)
        self.status_text.append(f"{os.path.basename(image_path)}: 检测到 {len(masks_np)} 个对象")
    
    def display_image(self, image_np):
        """
        在QLabel中显示numpy图像
        
        Args:
            image_np: numpy图像数组
        """
        # 将BGR转换为RGB（如果需要）
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 转换为QImage
        height, width, channel = image_np.shape
        bytes_per_line = 3 * width
        qimg = QImage(image_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 转换为QPixmap并缩放以适应预览区域
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # 显示图像
        self.preview_label.setPixmap(scaled_pixmap)
    
    def segmentation_finished(self, success, message):
        """
        分割完成
        """
        self.progress_bar.setValue(100)
        self.status_text.append(message)
        
        # 启用控件
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.path_button.setEnabled(True)
        self.output_button.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.radio_single.setEnabled(True)
        self.radio_folder.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "警告", message)

def main():
    """
    主函数
    """
    app = QApplication(sys.argv)
    window = SAM3SegmentationGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
