"""
FastAPI service for SAM3 ONNX inference.
Run:
  uvicorn api_server:app --host 0.0.0.0 --port 8000
"""

import json
import os
import re
import subprocess
import sys
import threading
import uuid
import zipfile
from dataclasses import dataclass
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


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_DEVICES = {"cuda", "cpu", "npu", "cann"}
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

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path("onnx-models").resolve()
TOKENIZER_PATH = (MODEL_DIR / "tokenizer.json").resolve()
OUTPUT_DIR = Path("segmentation_output/api").resolve()
MODEL_PRECISION = "fp32"
DEFAULT_DEVICE = "cuda"
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_BACKEND = "inference"
DEFAULT_NUM_PROCESSES = 4
MAX_NUM_PROCESSES = 32
MAX_ARCHIVE_IMAGES = 1000
MAX_ARCHIVE_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    tokenizer_path: Path
    output_dir: Path
    conf_threshold: float
    model_precision: str


SETTINGS = Settings(
    model_dir=MODEL_DIR,
    tokenizer_path=TOKENIZER_PATH,
    output_dir=OUTPUT_DIR,
    conf_threshold=DEFAULT_CONF_THRESHOLD,
    model_precision=MODEL_PRECISION,
)
app = FastAPI(title="SAM3 ONNX Detection API", version="1.0.0")
app.mount(
    "/outputs",
    StaticFiles(directory=str(SETTINGS.output_dir), check_dir=False),
    name="outputs",
)

cors_origins = os.getenv("SAM3_CORS_ORIGINS")
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in cors_origins.split(",") if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

_engines: dict[str, Sam3ONNXInference] = {}
_engine_lock = threading.Lock()
_predict_lock = threading.Lock()
_batch_lock = threading.Lock()


def _onnx_name(base_name: str) -> str:
    if SETTINGS.model_precision == "fp16":
        return f"{base_name}-fp16.onnx"
    if SETTINGS.model_precision != "fp32":
        raise RuntimeError("MODEL_PRECISION must be 'fp32' or 'fp16'.")
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


def get_engine(device: Optional[str] = None) -> Sam3ONNXInference:
    normalized_device = normalize_device(device)
    if normalized_device in _engines:
        return _engines[normalized_device]

    with _engine_lock:
        if normalized_device not in _engines:
            model_dir = SETTINGS.model_dir
            _engines[normalized_device] = Sam3ONNXInference(
                vision_encoder_path=_require_file(model_dir / _onnx_name("vision-encoder")),
                text_encoder_path=_require_file(model_dir / _onnx_name("text-encoder")),
                geometry_encoder_path=_require_file(model_dir / _onnx_name("geometry-encoder")),
                decoder_path=_require_file(model_dir / _onnx_name("decoder")),
                tokenizer_path=_require_file(SETTINGS.tokenizer_path),
                device=normalized_device,
            )
    return _engines[normalized_device]


def sanitize_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or "image"


def decode_image(image_bytes: bytes) -> np.ndarray:
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
    engine = get_engine(device)
    # Keep one GPU inference active at a time; this avoids pressure spikes on smaller cards.
    with _predict_lock:
        return engine.predict(
            image_rgb,
            text=text,
            boxes=boxes,
            box_labels=box_labels,
            conf_threshold=conf_threshold,
        )


def run_batch_backend(
    backend: str,
    input_dir: Path,
    output_dir: Path,
    text: str,
    conf_threshold: float,
    device: str,
    num_processes: int,
) -> None:
    script_path = PROJECT_DIR / f"{backend}.py"
    if not script_path.is_file():
        raise RuntimeError(f"Backend script not found: {script_path}")

    command = [
        sys.executable,
        str(script_path),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(output_dir),
        "--text",
        text,
        "--model_dir",
        str(SETTINGS.model_dir),
        "--tokenizer",
        str(SETTINGS.tokenizer_path),
        "--device",
        device,
        "--conf",
        str(conf_threshold),
    ]
    if backend == "parallel_batch_inference":
        command.extend(["--num_processes", str(num_processes)])

    # Batch scripts load their own ONNX sessions, so serialize script jobs to
    # avoid several requests exhausting the same accelerator at once.
    with _batch_lock:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            check=False,
        )

    if completed.returncode != 0:
        error_bytes = completed.stderr or completed.stdout
        error_output = (
            error_bytes.decode("utf-8", errors="replace").strip()
            if error_bytes
            else "Unknown error"
        )
        raise RuntimeError(f"{backend} failed: {error_output[-2000:]}")


async def save_uploaded_archive(upload: UploadFile, destination: Path) -> None:
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
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
            if member_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue

            total_uncompressed_bytes += member.file_size
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


def artifact_url(request_id: str, artifact_path: Path, job_dir: Path) -> str:
    relative_path = artifact_path.relative_to(job_dir).as_posix()
    return f"/outputs/{request_id}/{relative_path}"


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
    with open(upload_path, "wb") as output_file:
        output_file.write(uploaded_bytes)

    parsed_boxes = None
    box_labels = None
    if boxes:
        try:
            parsed_boxes, box_labels = parse_box_prompts(boxes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = await run_in_threadpool(
        run_prediction,
        image_rgb,
        text,
        parsed_boxes,
        box_labels,
        conf_threshold,
        device,
    )

    out_json = job_dir / f"{base_name}_result.json"
    save_labelme_json(
        str(upload_path),
        results.get("boxes"),
        results.get("scores"),
        text,
        str(out_json),
    )

    out_txt = None
    if save_coords:
        out_txt = job_dir / f"{base_name}_coords.txt"
        save_coords_txt(
            str(upload_path),
            results.get("boxes"),
            results.get("scores"),
            text,
            str(out_txt),
        )

    visualization_path = None
    if save_visualization:
        visualize_results(image_bgr, results, str(job_dir), str(upload_path))
        candidate = job_dir / f"{base_name}_output{suffix}"
        if candidate.is_file():
            visualization_path = candidate

    with open(out_json, "r", encoding="utf-8") as result_file:
        result_json = json.load(result_file)

    return {
        "request_id": request_id,
        "detections": len(result_json.get("shapes", [])),
        "backend": "inference",
        "device": device,
        "conf": conf_threshold,
        "num_processes": None,
        "result": result_json,
        "files": {
            "input_image": str(upload_path),
            "input_image_url": artifact_url(request_id, upload_path, job_dir),
            "result_json": str(out_json),
            "result_json_url": artifact_url(request_id, out_json, job_dir),
            "coords_txt": str(out_txt) if out_txt else None,
            "coords_txt_url": (
                artifact_url(request_id, out_txt, job_dir) if out_txt else None
            ),
            "visualization": str(visualization_path) if visualization_path else None,
            "visualization_url": (
                artifact_url(request_id, visualization_path, job_dir)
                if visualization_path
                else None
            ),
        },
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
    archive_name = f"{sanitize_stem(archive.filename or 'images')}.zip"
    archive_path = job_dir / archive_name
    input_dir = job_dir / "input"
    result_dir = job_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

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

    await run_in_threadpool(
        run_batch_backend,
        backend,
        input_dir,
        result_dir,
        text,
        conf_threshold,
        device,
        num_processes,
    )

    response_results = []
    failed_images = []
    total_detections = 0
    for relative_image_path in image_paths:
        input_path = input_dir / relative_image_path
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

        out_txt = generated_txt if save_coords and generated_txt.is_file() else None
        visualization_path = (
            generated_visualization
            if save_visualization and generated_visualization.is_file()
            else None
        )
        response_results.append(
            {
                "image": relative_image_path.as_posix(),
                "detections": detections,
                "result": result_json,
                "files": {
                    "input_image": str(input_path),
                    "input_image_url": artifact_url(request_id, input_path, job_dir),
                    "result_json": str(out_json),
                    "result_json_url": artifact_url(request_id, out_json, job_dir),
                    "coords_txt": str(out_txt) if out_txt else None,
                    "coords_txt_url": (
                        artifact_url(request_id, out_txt, job_dir) if out_txt else None
                    ),
                    "visualization": (
                        str(visualization_path) if visualization_path else None
                    ),
                    "visualization_url": (
                        artifact_url(request_id, visualization_path, job_dir)
                        if visualization_path
                        else None
                    ),
                },
            }
        )

    if not response_results:
        raise RuntimeError(f"{backend} did not produce a result JSON for any image.")

    return {
        "request_id": request_id,
        "backend": backend,
        "device": device,
        "conf": conf_threshold,
        "num_processes": (
            num_processes if backend == "parallel_batch_inference" else None
        ),
        "total_images": len(image_paths),
        "completed_images": len(response_results),
        "failed_images": failed_images,
        "detections": total_detections,
        "results": response_results,
        "files": {
            "input_archive": str(archive_path),
            "input_archive_url": artifact_url(request_id, archive_path, job_dir),
            "result_directory": str(result_dir),
        },
    }


@app.on_event("startup")
def startup() -> None:
    SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "loaded_devices": sorted(_engines.keys()),
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

    request_id = uuid.uuid4().hex
    job_dir = SETTINGS.output_dir / request_id
    job_dir.mkdir(parents=True, exist_ok=True)

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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
