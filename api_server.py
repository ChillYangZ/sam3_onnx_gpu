"""SAM3 ONNX 推理服务。

启动命令：
  uvicorn api_server:app --host 0.0.0.0 --port 8510
"""

import json
import logging
import gc
import os
import re
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from inference import (
    Sam3ONNXInference,
    parse_box_prompts,
    save_coords_txt,
    save_labelme_json,
    visualize_results,
)


# 客户端允许传入的图片、设备及后端名称。
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_DEVICES = {"cuda", "cpu", "npu", "cann"}
# 同时接受带 .py 和不带 .py 的后端名称，并统一为内部标准名称。
SUPPORTED_BACKENDS = {
    "inference": "inference",
    "inference.py": "inference",
    "batch_inference": "batch_inference",
    "batch_inference.py": "batch_inference",
    "fp16batch_inference": "FP16batch_inference",
    "fp16batch_inference.py": "FP16batch_inference",
    "parallel_batch_inference": "parallel_batch_inference",
    "parallel_batch_inference.py": "parallel_batch_inference",
}

# 以下为服务端固定配置，不接受 curl 或环境变量覆盖。
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path("onnx-models").resolve()
TOKENIZER_PATH = (MODEL_DIR / "tokenizer.json").resolve()
OUTPUT_DIR = Path("segmentation_output/api").resolve()
MODEL_PRECISION = "fp32"
SAVE_RESULTS_LOCALLY = True   # False 表示响应生成后删除本次请求目录，但推理期间仍会使用临时文件。

# curl 可配置参数及默认值
DEFAULT_DEVICE = "cuda"
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_BACKEND = "inference"
DEFAULT_NUM_PROCESSES = 4

# 上传限制用于防止超大压缩包耗尽磁盘或内存。
MAX_NUM_PROCESSES = 32
MAX_ARCHIVE_IMAGES = 1000
MAX_ARCHIVE_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class Settings:
    """服务启动后保持不变的配置快照。"""

    model_dir: Path
    tokenizer_path: Path
    output_dir: Path
    conf_threshold: float
    model_precision: str
    save_results_locally: bool


SETTINGS = Settings(
    model_dir=MODEL_DIR,
    tokenizer_path=TOKENIZER_PATH,
    output_dir=OUTPUT_DIR,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    model_precision=MODEL_PRECISION,
    save_results_locally=SAVE_RESULTS_LOCALLY,
)
app = FastAPI(title="SAM3 ONNX Detection API", version="1.0.0")
# 保留静态目录挂载以兼容已有部署；接口响应本身不会返回本地文件路径。
app.mount(
    "/outputs",
    StaticFiles(directory=str(SETTINGS.output_dir), check_dir=False),
    name="outputs",
)

cors_origins = os.getenv("SAM3_CORS_ORIGINS")
if cors_origins:
    # 仅 CORS 白名单保留环境变量入口，多个来源使用逗号分隔。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

# 模型缓存采用单槽位设计：切换精度或设备时先释放旧模型，避免显存中同时驻留多份模型。
_engines: dict[tuple[str, str], Sam3ONNXInference] = {}
# 模型切换和推理共用可重入锁，防止推理过程中 Session 被另一个请求释放。
_model_lock = threading.RLock()


def _onnx_name(base_name: str, precision: str) -> str:
    if precision == "fp16":
        return f"{base_name}-fp16.onnx"
    if precision != "fp32":
        raise RuntimeError("precision must be 'fp32' or 'fp16'.")
    return f"{base_name}.onnx"


def _require_file(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"Required file not found: {path}")
    return str(path)


def normalize_device(device: Optional[str]) -> str:
    normalized = (device or DEFAULT_DEVICE).strip().lower()
    if normalized not in SUPPORTED_DEVICES:
        raise ValueError(f"device must be one of: {', '.join(sorted(SUPPORTED_DEVICES))}")
    return normalized


def normalize_backend(backend: Optional[str]) -> str:
    normalized = (backend or DEFAULT_BACKEND).strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        supported = ", ".join(
            ["inference", "batch_inference", "FP16batch_inference", "parallel_batch_inference"]
        )
        raise ValueError(f"backend must be one of: {supported}")
    return SUPPORTED_BACKENDS[normalized]


def get_engine(
    device: Optional[str] = None,
    precision: str = MODEL_PRECISION,
) -> Sam3ONNXInference:
    """获取指定设备和精度的模型；缓存不匹配时释放旧模型并重新加载。"""

    normalized_device = normalize_device(device)
    normalized_precision = precision.strip().lower()
    if normalized_precision not in {"fp32", "fp16"}:
        raise ValueError("precision must be 'fp32' or 'fp16'.")
    engine_key = (normalized_precision, normalized_device)
    with _model_lock:
        if engine_key in _engines:
            return _engines[engine_key]

        if _engines:
            # 当前只允许一个模型槽位，因此任何设备或精度变化都会触发完整切换。
            previous_models = [
                f"{loaded_precision}/{loaded_device}"
                for loaded_precision, loaded_device in _engines.keys()
            ]
            logger.info(
                "Unloading model before switch: loaded=%s requested=%s/%s",
                ",".join(previous_models),
                normalized_precision,
                normalized_device,
            )
            _engines.clear()
            collected_objects = gc.collect()
            logger.info(
                "Previous model released: garbage_collected=%d",
                collected_objects,
            )

        load_started = time.perf_counter()
        logger.info(
            "Loading model: device=%s precision=%s",
            normalized_device,
            normalized_precision,
        )
        model_dir = SETTINGS.model_dir
        _engines[engine_key] = Sam3ONNXInference(
            vision_encoder_path=_require_file(
                model_dir / _onnx_name("vision-encoder", normalized_precision)
            ),
            text_encoder_path=_require_file(
                model_dir / _onnx_name("text-encoder", normalized_precision)
            ),
            geometry_encoder_path=_require_file(
                model_dir / _onnx_name("geometry-encoder", normalized_precision)
            ),
            decoder_path=_require_file(
                model_dir / _onnx_name("decoder", normalized_precision)
            ),
            tokenizer_path=_require_file(SETTINGS.tokenizer_path),
            device=normalized_device,
        )
        logger.info(
            "Model loaded: device=%s precision=%s elapsed=%.2fs",
            normalized_device,
            normalized_precision,
            time.perf_counter() - load_started,
        )
        return _engines[engine_key]


def sanitize_stem(filename: str) -> str:
    """清理客户端文件名，只保留可安全用于服务端文件名的字符。"""

    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or "image"


def create_request_job_dir() -> tuple[str, Path]:
    """按服务端收到请求的时间创建独立工作目录。"""

    while True:
        request_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        job_dir = SETTINGS.output_dir / request_id
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return request_id, job_dir


def cleanup_request_job_dir(request_id: str, job_dir: Path) -> None:
    """删除一次请求产生的上传文件、临时文件和推理结果。"""

    try:
        shutil.rmtree(job_dir)
    except FileNotFoundError:
        return
    except OSError:
        logger.exception(
            "[%s] Failed to clean temporary request directory: %s",
            request_id,
            job_dir,
        )
    else:
        logger.info("[%s] Temporary request directory removed: %s", request_id, job_dir)


def decode_image(image_bytes: bytes) -> np.ndarray:
    """将上传的图片字节解码为 OpenCV BGR 图像。"""

    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Uploaded file is not a readable image.")
    return image_bgr


def run_prediction(
    image_rgb: np.ndarray,
    text: Optional[str],
    boxes: Optional[list],
    box_labels: Optional[list],
    conf_threshold: float,
    device: str,
) -> dict:
    # 模型选择和推理必须处于同一个锁区间，避免其他请求在推理中途切换模型。
    with _model_lock:
        engine = get_engine(device, SETTINGS.model_precision)
        return engine.predict(
            image_rgb,
            text=text,
            boxes=boxes,
            box_labels=box_labels,
            conf_threshold=conf_threshold,
        )


def run_batch_backend(
    request_id: str,
    backend: str,
    input_dir: Path,
    output_dir: Path,
    text: str,
    conf_threshold: float,
    device: str,
    num_processes: int,
    image_paths: list[Path],
    save_visualization: bool,
    save_coords: bool,
) -> None:
    """在模型锁内执行整批任务，确保批处理中途不会发生模型切换。"""

    with _model_lock:
        _run_batch_backend_locked(
            request_id,
            backend,
            input_dir,
            output_dir,
            text,
            conf_threshold,
            device,
            num_processes,
            image_paths,
            save_visualization,
            save_coords,
        )


def _run_batch_backend_locked(
    request_id: str,
    backend: str,
    input_dir: Path,
    output_dir: Path,
    text: str,
    conf_threshold: float,
    device: str,
    num_processes: int,
    image_paths: list[Path],
    save_visualization: bool,
    save_coords: bool,
) -> None:
    """执行已持有模型锁的批处理核心逻辑。"""

    # 只有 FP16 批处理后端使用半精度，其余后端均使用 FP32。
    precision = "fp16" if backend == "FP16batch_inference" else "fp32"
    engine = get_engine(device, precision)
    # 顺序后端固定单线程；并行后端的线程数不会超过图片数量。
    worker_count = (
        min(num_processes, len(image_paths))
        if backend == "parallel_batch_inference"
        else 1
    )

    logger.info(
        "[%s] Batch backend starting: backend=%s device=%s precision=%s conf=%.4f workers=%d model_cache=ready",
        request_id,
        backend,
        device,
        precision,
        conf_threshold,
        worker_count,
    )

    def process_image(relative_path: Path) -> None:
        """处理一张图片；保留相对目录结构以避免同名图片互相覆盖。"""

        image_path = input_dir / relative_path
        output_subdir = output_dir / relative_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        image_started = time.perf_counter()
        try:
            image_bgr = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image_bgr is None:
                raise ValueError("Image is not readable.")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = engine.predict(
                image_rgb,
                text=text,
                conf_threshold=conf_threshold,
            )

            out_json = output_subdir / f"{relative_path.stem}_result.json"
            save_labelme_json(
                str(image_path),
                results.get("boxes"),
                results.get("scores"),
                text,
                str(out_json),
            )
            if save_coords:
                save_coords_txt(
                    str(image_path),
                    results.get("boxes"),
                    results.get("scores"),
                    text,
                    str(output_subdir / f"{relative_path.stem}_coords.txt"),
                )
            if save_visualization:
                visualize_results(image_rgb, results, str(output_subdir), str(image_path))

            boxes = results.get("boxes")
            detections = len(boxes) if boxes is not None else 0
            logger.info(
                "[%s][%s] image=%s detections=%d elapsed=%.3fs",
                request_id,
                backend,
                relative_path.as_posix(),
                detections,
                time.perf_counter() - image_started,
            )
        except Exception as exc:
            logger.exception(
                "[%s][%s] image failed: image=%s error=%s",
                request_id,
                backend,
                relative_path.as_posix(),
                exc,
            )

    if worker_count == 1:
        for image_path in image_paths:
            process_image(image_path)
    else:
        # 多个线程共享同一个 ONNX Session，不会为每个 worker 重复加载模型。
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(process_image, path) for path in image_paths]
            for future in as_completed(futures):
                future.result()

    logger.info("[%s] Batch backend completed: backend=%s", request_id, backend)


async def save_uploaded_archive(upload: UploadFile, destination: Path) -> None:
    """分块保存上传的 ZIP，并在写入过程中执行大小限制。"""

    total_bytes = 0
    with open(destination, "wb") as output_file:
        while chunk := await upload.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_ARCHIVE_UPLOAD_BYTES:
                raise ValueError(
                    f"Archive exceeds the {MAX_ARCHIVE_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                )
            output_file.write(chunk)
    if total_bytes == 0:
        raise ValueError("Uploaded archive is empty.")


def extract_image_archive(archive_path: Path, input_dir: Path) -> list[Path]:
    """安全解压 ZIP 中受支持的图片，并返回相对于输入目录的路径。"""

    if not zipfile.is_zipfile(archive_path):
        raise ValueError("Batch input must be a valid ZIP archive.")

    input_dir.mkdir(parents=True, exist_ok=True)
    input_root = input_dir.resolve()
    image_paths: list[Path] = []
    extracted_targets: set[Path] = set()
    result_keys: set[str] = set()
    total_uncompressed_bytes = 0
    extracted_bytes = 0

    with zipfile.ZipFile(archive_path) as archive_file:
        for member in archive_file.infolist():
            member_path = PurePosixPath(member.filename.replace("\\", "/"))
            if member.is_dir():
                continue
            # 拒绝绝对路径和 ..，防止 ZIP 路径穿越到请求目录之外。
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
            if member_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue

            total_uncompressed_bytes += member.file_size
            # 同时检查 ZIP 元数据和实际解压字节数，降低 ZIP 炸弹风险。
            if total_uncompressed_bytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                raise ValueError(
                    "Uncompressed images exceed the "
                    f"{MAX_ARCHIVE_UNCOMPRESSED_BYTES // (1024 * 1024)} MB limit."
                )
            if len(image_paths) >= MAX_ARCHIVE_IMAGES:
                raise ValueError(
                    f"ZIP archive contains more than {MAX_ARCHIVE_IMAGES} images."
                )

            relative_path = Path(*member_path.parts)
            destination = (input_root / relative_path).resolve()
            try:
                destination.relative_to(input_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}") from exc
            if destination in extracted_targets:
                raise ValueError(f"Duplicate image path in ZIP archive: {member.filename}")
            # 相同目录下不同扩展名但同 stem 的图片会生成同名结果，需提前拒绝。
            result_key = (relative_path.parent / relative_path.stem).as_posix().lower()
            if result_key in result_keys:
                raise ValueError(
                    "Images with the same folder and stem would overwrite results: "
                    f"{member.filename}"
                )

            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive_file.open(member) as source, open(destination, "wb") as output_file:
                while chunk := source.read(1024 * 1024):
                    extracted_bytes += len(chunk)
                    if extracted_bytes > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
                        raise ValueError(
                            "Extracted images exceed the "
                            f"{MAX_ARCHIVE_UNCOMPRESSED_BYTES // (1024 * 1024)} MB limit."
                        )
                    output_file.write(chunk)

            extracted_targets.add(destination)
            result_keys.add(result_key)
            image_paths.append(relative_path)

    if not image_paths:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(f"ZIP archive contains no supported images ({supported}).")
    return sorted(image_paths)


async def run_single_api_request(
    image: UploadFile,
    request_id: str,
    job_dir: Path,
    text: Optional[str],
    boxes: Optional[str],
    conf_threshold: float,
    device: str,
    save_visualization: bool,
    save_coords: bool,
) -> dict:
    """处理单图请求并组装可直接返回给客户端的 LabelMe JSON。"""

    request_started = time.perf_counter()
    uploaded_bytes = await image.read()
    if not uploaded_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    original_name = Path(image.filename or "image.jpg").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        suffix = ".jpg"

    try:
        image_bgr = decode_image(uploaded_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    base_name = sanitize_stem(original_name)
    upload_path = job_dir / f"{base_name}{suffix}"
    # 保存原图供 LabelMe 结果和可视化函数引用；关闭持久化时会在 finally 中统一删除。
    with open(upload_path, "wb") as output_file:
        output_file.write(uploaded_bytes)

    logger.info(
        "[%s] Image received: file=%s size=%d bytes resolution=%dx%d",
        request_id,
        original_name,
        len(uploaded_bytes),
        image_bgr.shape[1],
        image_bgr.shape[0],
    )

    parsed_boxes = None
    box_labels = None
    if boxes:
        try:
            parsed_boxes, box_labels = parse_box_prompts(boxes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inference_started = time.perf_counter()
    logger.info(
        "[%s] Inference started: backend=inference device=%s conf=%.4f text_prompt=%s box_prompts=%d",
        request_id,
        device,
        conf_threshold,
        "yes" if text else "no",
        len(parsed_boxes) if parsed_boxes else 0,
    )
    results = await run_in_threadpool(
        run_prediction,
        image_rgb,
        text,
        parsed_boxes,
        box_labels,
        conf_threshold,
        device,
    )
    inference_elapsed = time.perf_counter() - inference_started

    # 现有 LabelMe 工具以文件为输出，因此先落盘再读取为响应对象。
    out_json = job_dir / f"{base_name}_result.json"
    save_labelme_json(
        str(upload_path),
        results.get("boxes"),
        results.get("scores"),
        text,
        str(out_json),
    )

    if save_coords:
        out_txt = job_dir / f"{base_name}_coords.txt"
        save_coords_txt(
            str(upload_path),
            results.get("boxes"),
            results.get("scores"),
            text,
            str(out_txt),
        )

    if save_visualization:
        visualize_results(image_bgr, results, str(job_dir), str(upload_path))

    with open(out_json, "r", encoding="utf-8") as result_file:
        result_json = json.load(result_file)

    detection_count = len(result_json.get("shapes", []))
    logger.info(
        "[%s] Inference completed: detections=%d inference_elapsed=%.3fs total_elapsed=%.3fs output_dir=%s",
        request_id,
        detection_count,
        inference_elapsed,
        time.perf_counter() - request_started,
        job_dir,
    )

    return {
        "request_id": request_id,
        "backend": "inference",
        "device": device,
        "conf": conf_threshold,
        "num_processes": None,
        "detections": detection_count,
        "result": result_json,
    }


async def run_batch_api_request(
    archive: UploadFile,
    request_id: str,
    job_dir: Path,
    backend: str,
    text: str,
    conf_threshold: float,
    device: str,
    num_processes: int,
    save_visualization: bool,
    save_coords: bool,
) -> dict:
    """保存并解压 ZIP，执行批处理后汇总每张图片的 JSON 结果。"""

    request_started = time.perf_counter()
    archive_name = f"{sanitize_stem(archive.filename or 'images')}.zip"
    archive_path = job_dir / archive_name
    input_dir = job_dir / "input"
    result_dir = job_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # ZIP 保存和解压都可能涉及较多磁盘 I/O，解压工作放在线程池中执行。
    try:
        await save_uploaded_archive(archive, archive_path)
        image_paths = await run_in_threadpool(
            extract_image_archive,
            archive_path,
            input_dir,
        )
    except (
        NotImplementedError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "[%s] Archive received and extracted: file=%s images=%d backend=%s",
        request_id,
        archive.filename or "images.zip",
        len(image_paths),
        backend,
    )

    await run_in_threadpool(
        run_batch_backend,
        request_id,
        backend,
        input_dir,
        result_dir,
        text,
        conf_threshold,
        device,
        num_processes,
        image_paths,
        save_visualization,
        save_coords,
    )

    response_results = []
    failed_images = []
    total_detections = 0
    # 批处理核心将结果写入文件，此处统一读回内存并组成 HTTP 响应。
    for relative_image_path in image_paths:
        output_subdir = result_dir / relative_image_path.parent
        out_json = output_subdir / f"{relative_image_path.stem}_result.json"
        generated_txt = output_subdir / f"{relative_image_path.stem}_coords.txt"
        generated_visualization = (
            output_subdir
            / f"{relative_image_path.stem}_output{relative_image_path.suffix}"
        )

        if not out_json.is_file():
            failed_images.append(relative_image_path.as_posix())
            continue
        if not save_coords and generated_txt.is_file():
            generated_txt.unlink()
        if not save_visualization and generated_visualization.is_file():
            generated_visualization.unlink()

        with open(out_json, "r", encoding="utf-8") as result_file:
            result_json = json.load(result_file)
        detections = len(result_json.get("shapes", []))
        total_detections += detections
        logger.info(
            "[%s] Result collected: image=%s detections=%d",
            request_id,
            relative_image_path.as_posix(),
            detections,
        )

        response_results.append(
            {
                "image": relative_image_path.as_posix(),
                "detections": detections,
                "result": result_json,
            }
        )

    if not response_results:
        raise RuntimeError(f"{backend} did not produce a result JSON for any image.")

    logger.info(
        "[%s] Batch request completed: completed=%d total=%d failed=%d detections=%d elapsed=%.3fs output_dir=%s",
        request_id,
        len(response_results),
        len(image_paths),
        len(failed_images),
        total_detections,
        time.perf_counter() - request_started,
        result_dir,
    )

    return {
        "request_id": request_id,
        "backend": backend,
        "device": device,
        "conf": conf_threshold,
        "num_processes": (
            num_processes if backend == "parallel_batch_inference" else None
        ),
        "detections": total_detections,
        "total_images": len(image_paths),
        "completed_images": len(response_results),
        "failed_images": failed_images,
        "results": response_results,
    }


@app.on_event("startup")
def startup() -> None:
    """创建输出根目录，并在服务开始接收请求前预加载默认 FP32 模型。"""

    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "SAM3 API starting: preloading precision=%s device=%s "
        "output_dir=%s save_results_locally=%s",
        SETTINGS.model_precision,
        DEFAULT_DEVICE,
        SETTINGS.output_dir,
        SETTINGS.save_results_locally,
    )
    get_engine(DEFAULT_DEVICE, SETTINGS.model_precision)
    logger.info(
        "SAM3 API ready: default_backend=%s default_device=%s precision=%s",
        DEFAULT_BACKEND,
        DEFAULT_DEVICE,
        SETTINGS.model_precision,
    )


@app.get("/health")
def health() -> dict:
    """返回服务配置和当前唯一模型槽位的加载状态，不执行推理。"""

    loaded_models = [
        {"precision": precision, "device": device}
        for precision, device in sorted(_engines.keys())
    ]
    return {
        "status": "ok",
        "loaded_devices": sorted({device for _, device in _engines.keys()}),
        "loaded_models": loaded_models,
        "default_device": DEFAULT_DEVICE,
        "supported_devices": sorted(SUPPORTED_DEVICES),
        "default_conf_threshold": SETTINGS.conf_threshold,
        "default_backend": DEFAULT_BACKEND,
        "supported_backends": [
            "inference",
            "batch_inference",
            "FP16batch_inference",
            "parallel_batch_inference",
        ],
        "default_num_processes": DEFAULT_NUM_PROCESSES,
        "model_precision": SETTINGS.model_precision,
        "model_dir": str(SETTINGS.model_dir),
        "tokenizer_path": str(SETTINGS.tokenizer_path),
        "output_dir": str(SETTINGS.output_dir),
    }


@app.post("/predict")
async def predict(
    image: Optional[UploadFile] = File(
        None,
        description="Single image for inference, or a ZIP alias for batch backends.",
    ),
    archive: Optional[UploadFile] = File(
        None,
        description="ZIP archive containing an image folder for batch backends.",
    ),
    text: Optional[str] = Form(None, description="Text prompt, for example: person"),
    prompt: Optional[str] = Form(None, description="Alias for text."),
    boxes: Optional[str] = Form(None, description="Optional box prompts: pos:x,y,w,h;neg:x,y,w,h in xywh pixels."),
    conf: Optional[float] = Form(None, description="Confidence threshold."),
    device: str = Form(DEFAULT_DEVICE, description="Inference device: cuda, cpu, npu, or cann."),
    backend: str = Form(
        DEFAULT_BACKEND,
        description="inference, batch_inference, FP16batch_inference, or parallel_batch_inference.",
    ),
    num_processes: int = Form(
        DEFAULT_NUM_PROCESSES,
        description="Worker processes used by parallel_batch_inference.",
    ),
    save_visualization: bool = Form(True),
    save_coords: bool = Form(True),
) -> dict:
    """校验 multipart/form-data 参数，并将请求分派到单图或批处理流程。"""

    # text 是标准字段，prompt 作为兼容别名；两者同时存在时优先使用 text。
    effective_text = text if text is not None else prompt
    if effective_text is not None:
        effective_text = effective_text.strip()
    if not effective_text:
        effective_text = None

    try:
        effective_device = normalize_device(device)
        effective_backend = normalize_backend(backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 单图与批处理后端接受的上传字段和提示词类型不同，需分别校验。
    if effective_backend == "inference":
        if not effective_text and not boxes:
            raise HTTPException(status_code=400, detail="Please provide text/prompt or boxes.")
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="backend=inference requires an image upload.",
            )
        if archive is not None:
            raise HTTPException(
                status_code=400,
                detail="backend=inference does not accept archive.",
            )
    else:
        if not effective_text:
            raise HTTPException(
                status_code=400,
                detail=f"{effective_backend} requires text/prompt.",
            )
        if boxes:
            raise HTTPException(
                status_code=400,
                detail=f"{effective_backend} does not support box prompts.",
            )
        if archive is not None and image is not None:
            raise HTTPException(
                status_code=400,
                detail="Provide only archive, not both archive and image.",
            )
        if archive is None:
            if image is not None and Path(image.filename or "").suffix.lower() == ".zip":
                archive = image
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"{effective_backend} requires a ZIP upload in archive.",
                )

    # curl 未传 conf 时使用服务端默认值，显式传入时以请求值为准。
    effective_conf = SETTINGS.conf_threshold if conf is None else conf
    if not 0 <= effective_conf <= 1:
        raise HTTPException(status_code=400, detail="conf must be between 0 and 1.")
    if (
        effective_backend == "parallel_batch_inference"
        and not 1 <= num_processes <= MAX_NUM_PROCESSES
    ):
        raise HTTPException(
            status_code=400,
            detail=f"num_processes must be between 1 and {MAX_NUM_PROCESSES}.",
        )

    # 每次请求使用独立目录，避免并发请求之间的文件互相覆盖。
    request_id, job_dir = create_request_job_dir()

    logger.info(
        "[%s] Request accepted: backend=%s device=%s conf=%.4f",
        request_id,
        effective_backend,
        effective_device,
        effective_conf,
    )

    try:
        if effective_backend == "inference":
            return await run_single_api_request(
                image,
                request_id,
                job_dir,
                effective_text,
                boxes,
                effective_conf,
                effective_device,
                save_visualization,
                save_coords,
            )
        return await run_batch_api_request(
            archive,
            request_id,
            job_dir,
            effective_backend,
            effective_text,
            effective_conf,
            effective_device,
            num_processes,
            save_visualization,
            save_coords,
        )
    except HTTPException as exc:
        logger.warning("[%s] Request rejected: %s", request_id, exc.detail)
        raise
    except Exception as exc:
        logger.exception("[%s] Request failed: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # 响应数据此时已全部读入内存，因此删除目录不会影响返回的 JSON。
        if not SETTINGS.save_results_locally:
            await run_in_threadpool(cleanup_request_job_dir, request_id, job_dir)