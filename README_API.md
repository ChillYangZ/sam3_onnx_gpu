# SAM3 ONNX FastAPI Service

This project now includes `api_server.py`, a FastAPI service that accepts an
image plus a text prompt and returns the saved LabelMe JSON result content.

## Install API dependencies

```bash
pip install -r requirements-api.txt
```

If you deploy on CPU instead of CUDA, replace `onnxruntime-gpu` in
`requirements-api.txt` with `onnxruntime`.

## Start the service

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

By default it uses:

- model directory: `onnx-models`
- tokenizer: `onnx-models/tokenizer.json`
- output directory: `segmentation_output/api`
- confidence threshold: `0.5`
- model precision: `fp32`
- device: `cuda`

The model directory, tokenizer path, output directory, and model precision are
fixed in `api_server.py`; they are not read from `set` or `export`
environment variables. Change these constants in code if needed:

```python
MODEL_DIR = Path("onnx-models").resolve()
TOKENIZER_PATH = (MODEL_DIR / "tokenizer.json").resolve()
OUTPUT_DIR = Path("segmentation_output/api").resolve()
MODEL_PRECISION = "fp32"
```

`backend`, `device`, `conf`, and `num_processes` can be passed in each
`/predict` request. Their code defaults are:

- `backend=inference`
- `device=cuda`
- `conf=0.5`
- `num_processes=4` (only used by `parallel_batch_inference`)

Supported `backend` values:

- `inference`: call the inference implementation directly with FP32 models
- `batch_inference`: call `batch_inference.py` with FP32 models
- `FP16batch_inference`: call `FP16batch_inference.py` with FP16 models
- `parallel_batch_inference`: call `parallel_batch_inference.py` with FP32
  models and `num_processes` workers

The three batch backends currently require a text prompt and do not support
`boxes`. The default `inference` backend continues to support text and/or box
prompts. A remote HTTP client cannot upload a folder path directly, so batch
requests upload the image folder as a ZIP archive in the `archive` field. The
server extracts it into the request directory and preserves subfolders.

## Health check

```bash
curl http://127.0.0.1:8000/health
```

## Predict

```bash
curl.exe -X POST http://127.0.0.1:8000/predict ^
  -F "image=@segmentation_input/test.jpg" ^
  -F "text=person" ^
  -F "backend=inference" ^
  -F "device=cuda" ^
  -F "conf=0.5"
```

On Linux servers:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "image=@segmentation_input/test.jpg" \
  -F "text=person" \
  -F "backend=inference" \
  -F "device=cuda" \
  -F "conf=0.5"
```

Call `batch_inference.py`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "archive=@segmentation_input.zip" \
  -F "text=person" \
  -F "backend=batch_inference" \
  -F "device=cuda" \
  -F "conf=0.5"
```

Call `FP16batch_inference.py`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "archive=@segmentation_input.zip" \
  -F "text=person" \
  -F "backend=FP16batch_inference" \
  -F "device=cuda" \
  -F "conf=0.5"
```

Call `parallel_batch_inference.py`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "archive=@segmentation_input.zip" \
  -F "text=person" \
  -F "backend=parallel_batch_inference" \
  -F "num_processes=4" \
  -F "device=cuda" \
  -F "conf=0.5"
```

## Response JSON fields

All paths without a `_url` suffix are local absolute paths on the API server.
Fields ending in `_url` are relative HTTP URLs. For example, when the API base
URL is `http://127.0.0.1:8000`, a returned URL such as
`/outputs/abc/result.json` can be downloaded from
`http://127.0.0.1:8000/outputs/abc/result.json`.

### Common top-level fields

| Field | Type | Description |
| --- | --- | --- |
| `request_id` | string | Unique ID generated for this request. It is also the name of the request output directory. |
| `backend` | string | Backend that actually handled the request: `inference`, `batch_inference`, `FP16batch_inference`, or `parallel_batch_inference`. |
| `device` | string | Normalized inference device requested by the client, such as `cuda`, `cpu`, `npu`, or `cann`. ONNX Runtime may still use a configured fallback provider when the requested provider is unavailable. |
| `conf` | number | Effective confidence threshold. Only detections with a score greater than this value are retained. |
| `num_processes` | integer or null | Worker count for `parallel_batch_inference`. It is `null` for all other backends. |
| `detections` | integer | Number of detected shapes. For single-image inference this is the shape count for that image; for batch inference this is the sum across all successfully processed images. |
| `files` | object | Request-level saved file information. Its fields differ between single-image and batch responses, as described below. |

### Single-image `inference` fields

| Field | Type | Description |
| --- | --- | --- |
| `result` | object | LabelMe JSON content saved for the uploaded image. See "LabelMe result fields" below. |
| `files.input_image` | string | Local server path of the saved uploaded image. |
| `files.input_image_url` | string | Relative HTTP URL for the saved uploaded image. |
| `files.result_json` | string | Local server path of the saved LabelMe JSON file. |
| `files.result_json_url` | string | Relative HTTP URL for the saved LabelMe JSON file. |
| `files.coords_txt` | string or null | Local server path of the coordinate TXT file. It is `null` when `save_coords=false`. |
| `files.coords_txt_url` | string or null | Relative HTTP URL for the coordinate TXT file. It is `null` when `save_coords=false`. |
| `files.visualization` | string or null | Local server path of the visualization image. It is `null` when `save_visualization=false` or the image was not created. |
| `files.visualization_url` | string or null | Relative HTTP URL for the visualization image. It is `null` when no visualization file is available. |

### Batch response fields

These fields are returned by `batch_inference`, `FP16batch_inference`, and
`parallel_batch_inference`.

| Field | Type | Description |
| --- | --- | --- |
| `total_images` | integer | Number of supported image files found in the uploaded ZIP archive. |
| `completed_images` | integer | Number of images for which a result JSON file was successfully created. |
| `failed_images` | array of strings | ZIP-relative paths of images that did not produce a result JSON. It is an empty array when every image succeeded. |
| `results` | array of objects | One result item for every successfully processed image. Failed images are not included here. |
| `files.input_archive` | string | Local server path of the saved uploaded ZIP archive. |
| `files.input_archive_url` | string | Relative HTTP URL for downloading the saved uploaded ZIP archive. |
| `files.result_directory` | string | Local server directory containing all generated batch results. Subdirectories from the ZIP are preserved. |

Each object in the top-level `results` array contains:

| Field | Type | Description |
| --- | --- | --- |
| `results[].image` | string | Image path relative to the root of the uploaded ZIP, for example `camera_1/001.jpg`. |
| `results[].detections` | integer | Number of detected shapes for this image. |
| `results[].result` | object | LabelMe JSON content saved for this image. |
| `results[].files` | object | Saved input and output file information for this image. |
| `results[].files.input_image` | string | Local server path of the extracted input image. |
| `results[].files.input_image_url` | string | Relative HTTP URL for the extracted input image. |
| `results[].files.result_json` | string | Local server path of this image's saved LabelMe JSON file. |
| `results[].files.result_json_url` | string | Relative HTTP URL for this image's LabelMe JSON file. |
| `results[].files.coords_txt` | string or null | Local server path of this image's coordinate TXT file, or `null` when `save_coords=false`. |
| `results[].files.coords_txt_url` | string or null | Relative HTTP URL for this image's coordinate TXT file, or `null`. |
| `results[].files.visualization` | string or null | Local server path of this image's visualization, or `null` when unavailable. |
| `results[].files.visualization_url` | string or null | Relative HTTP URL for this image's visualization, or `null`. |

### LabelMe result fields

The single-image `result` object and every batch `results[].result` object use
the same LabelMe structure:

| Field | Type | Description |
| --- | --- | --- |
| `version` | string | LabelMe format version written by the inference code, currently `5.2.1`. |
| `flags` | object | LabelMe image-level flags. It is currently an empty object. |
| `shapes` | array of objects | Retained detections after applying `conf`. It is an empty array when nothing passes the threshold. |
| `imagePath` | string | Original image file name stored in the LabelMe JSON. |
| `imageData` | string or null | Embedded image data. The API does not embed images, so this is currently `null`. |
| `imageHeight` | integer | Source image height in pixels. |
| `imageWidth` | integer | Source image width in pixels. |

Each detection in `shapes` contains:

| Field | Type | Description |
| --- | --- | --- |
| `shapes[].label` | string | Text prompt used for the detection. For box-only single-image inference, the fallback label is `object`. |
| `shapes[].points` | array | Two pixel coordinates in `[[x1, y1], [x2, y2]]` format: top-left and bottom-right corners of the rectangle. |
| `shapes[].group_id` | integer or null | Optional LabelMe group ID. It is currently `null`. |
| `shapes[].description` | string | Detection confidence in a string such as `score: 0.9234`. |
| `shapes[].shape_type` | string | LabelMe shape type, currently `rectangle`. |
| `shapes[].flags` | object | LabelMe shape-level flags. It is currently an empty object. |

### Error response fields

For API errors such as an invalid backend, missing upload, invalid ZIP, or
out-of-range parameter, the response contains:

| Field | Type | Description |
| --- | --- | --- |
| `detail` | string or array | Human-readable error text. FastAPI request-schema validation errors may return an array of structured validation details instead of a string. |

Create the ZIP archive before sending it. PowerShell:

```powershell
Compress-Archive -Path .\segmentation_input\* `
  -DestinationPath .\segmentation_input.zip -Force
```

Linux:

```bash
cd segmentation_input
zip -r ../segmentation_input.zip .
```

Do not send a client path such as `input_dir=C:\images`; after deployment that
path refers to the server, not the computer running curl.

Optional box prompts are also supported:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -F "image=@segmentation_input/test.jpg" ^
  -F "text=person" ^
  -F "boxes=pos:5384,2352,397,785" ^
  -F "device=cuda" ^
  -F "conf=0.5"
```

## Shutdown the service

just press "CTRL + C"
