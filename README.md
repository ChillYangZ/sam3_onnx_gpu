# SAM3 ONNX FastAPI 服务

本项目提供 `api_server.py`，用于启动 SAM3 ONNX 的 FastAPI 服务。客户端可以上传单张图片或包含多张图片的 ZIP 压缩包，并通过文本提示词执行检测。接口会返回保存后的 LabelMe JSON 结果内容。
> 项目参考 [xiongmao6/ascend_sam3](https://github.com/xiongmao6/ascend_sam3)

## 安装 API 依赖

```bash
pip install -r requirements-api.txt
```

如果部署环境只使用 CPU，请将 `requirements-api.txt` 中的 `onnxruntime-gpu` 替换为 `onnxruntime`。

## 模型下载

导出的 SAM3 ONNX 模型可从以下链接获取：
> **[SAM3_ONNX Model下载](https://github.com/jamjamjon/assets/releases/tag/sam3)**

模型文件需要放在项目根目录的 `onnx-models` 文件夹中，并保持以下文件名不变。

### 需要下载的文件

| 精度 | 文件名 | 用途 |
| --- | --- | --- |
| FP32 | `vision-encoder.onnx` | 图片特征编码 |
| FP32 | `text-encoder.onnx` | 文本提示词编码 |
| FP32 | `geometry-encoder.onnx` | 框提示词编码 |
| FP32 | `decoder.onnx` | 检测结果解码 |
| FP16 | `vision-encoder-fp16.onnx` | FP16 图片特征编码 |
| FP16 | `text-encoder-fp16.onnx` | FP16 文本提示词编码 |
| FP16 | `geometry-encoder-fp16.onnx` | FP16 框提示词编码 |
| FP16 | `decoder-fp16.onnx` | FP16 检测结果解码 |
| 公共文件 | `tokenizer.json` | FP32 和 FP16 共用的文本分词器 |

服务启动时默认加载 FP32 模型，因此 4 个 FP32 模型和 `tokenizer.json` 必须下载。只有需要调用 `FP16batch_inference` 时，才需要额外下载 4 个 FP16 模型。

### Windows CMD 下载命令

在项目根目录打开 Windows CMD，然后执行：

```cmd
if not exist "onnx-models" mkdir "onnx-models"

curl.exe -L --fail --retry 3 -C - ^
  -o "onnx-models\vision-encoder.onnx" ^
  "https://github.com/jamjamjon/assets/releases/download/sam3/vision-encoder.onnx"
curl.exe -L --fail --retry 3 -C - ^
  -o "onnx-models\text-encoder.onnx" ^
  "https://github.com/jamjamjon/assets/releases/download/sam3/text-encoder.onnx"
curl.exe -L --fail --retry 3 -C - ^
  -o "onnx-models\geometry-encoder.onnx" ^
  "https://github.com/jamjamjon/assets/releases/download/sam3/geometry-encoder.onnx"
curl.exe -L --fail --retry 3 -C - ^
  -o "onnx-models\decoder.onnx" ^
  "https://github.com/jamjamjon/assets/releases/download/sam3/decoder.onnx"
curl.exe -L --fail --retry 3 -C - ^
  -o "onnx-models\tokenizer.json" ^
  "https://github.com/jamjamjon/assets/releases/download/sam3/tokenizer.json"

curl.exe -L --fail --retry 3 -C - ^
  -o "onnx-models\vision-encoder-fp16.onnx" ^
  "https://github.com/jamjamjon/assets/releases/download/sam3/vision-encoder-fp16.onnx"
curl.exe -L --fail --retry 3 -C - ^
  -o "onnx-models\text-encoder-fp16.onnx" ^
  "https://github.com/jamjamjon/assets/releases/download/sam3/text-encoder-fp16.onnx"
curl.exe -L --fail --retry 3 -C - ^
  -o "onnx-models\geometry-encoder-fp16.onnx" ^
  "https://github.com/jamjamjon/assets/releases/download/sam3/geometry-encoder-fp16.onnx"
curl.exe -L --fail --retry 3 -C - ^
  -o "onnx-models\decoder-fp16.onnx" ^
  "https://github.com/jamjamjon/assets/releases/download/sam3/decoder-fp16.onnx"
```

其中 `-L` 用于跟随 GitHub 下载重定向，`--retry 3` 表示失败后最多重试 3 次，`-C -` 支持大文件断点续传。下载完成后，`onnx-models` 文件夹中应包含上表列出的文件。

## 启动服务

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8510
```

默认配置如下：

- model directory: `onnx-models`
- tokenizer path: `onnx-models/tokenizer.json`
- output directory: `segmentation_output/api`
- confidence threshold: `0.5`
- model precision: `fp32`
- device: `cuda`
- save results locally: `True`

model directory、tokenizer path、output directory、model precision和结果保存开关固定配置在 `api_server.py` 中，不会读取通过 `set` 或 `export` 设置的环境变量。如需调整，请直接修改以下常量：

```python
MODEL_DIR = Path("onnx-models").resolve()
TOKENIZER_PATH = (MODEL_DIR / "tokenizer.json").resolve()
OUTPUT_DIR = Path("segmentation_output/api").resolve()
MODEL_PRECISION = "fp32"
SAVE_RESULTS_LOCALLY = True
```

将 `SAVE_RESULTS_LOCALLY` 修改为 `False` 后，请求处理期间仍会使用临时工作目录；响应 JSON 生成后，服务会删除该请求的整个目录，包括上传文件、解压图片、结果 JSON、坐标 TXT 和可视化图片。该配置不会改变接口返回的 JSON，也不能通过 curl 修改。

每次请求都可以设置 `backend`、`device`、`conf` 和 `num_processes`。代码中的默认值如下：

- `backend=inference`
- `device=cuda`
- `conf=0.5`
- `num_processes=4`，仅供 `parallel_batch_inference` 使用

支持的 `backend`：

- `inference`：直接调用单图推理实现，使用 FP32 模型。
- `batch_inference`：使用顺序批处理模式和缓存的 FP32 模型。
- `FP16batch_inference`：使用顺序批处理模式和缓存的 FP16 模型。
- `parallel_batch_inference`：使用 FP32 模型，并通过 `num_processes` 指定共享模型的并发线程数。

### 模型加载与缓存

- Uvicorn 启动时会直接加载默认设备上的 FP32 模型。只有模型加载成功后，服务才进入可用状态。
- 服务中任意时刻只保留一组模型，不会让 FP32 和 FP16 同时驻留内存或显存。
- `inference`、`batch_inference` 和 `parallel_batch_inference` 复用已加载的 FP32 模型。
- 收到 `FP16batch_inference` 请求时，服务会等待正在执行的推理完成，然后释放 FP32 Session、执行垃圾回收，再加载 FP16。
- FP16 已加载时，连续的 FP16 请求会直接复用；之后如果收到 FP32 请求，则释放 FP16 并重新加载 FP32。
- 设备变化也会触发同样的单槽位切换，例如从 `cuda` 切换为 `cpu` 时会先释放当前模型。
- `parallel_batch_inference` 在 API 进程内使用线程 worker 共享同一组 ONNX Session，`num_processes` 字段为了兼容现有接口而保留，实际表示并发线程数量。

精度或设备发生切换的第一次请求需要承担模型重新加载时间。为了减少切换开销，建议尽量连续发送相同精度和设备的请求。

GPU 部署建议只启动一个 Uvicorn worker：

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8510 --workers 1
```

如果使用 `--workers 2` 或更多，每个 Uvicorn 进程都有自己的模型槽位，仍会在显存中出现多份模型。

三个批处理后端必须提供文本提示词，并且不支持 `boxes`。默认的 `inference` 后端支持文本提示词、框提示词或同时使用两者。

远程 HTTP 客户端不能直接上传本地文件夹路径。因此，批处理请求需要先将图片文件夹压缩为 ZIP，再通过 `archive` 字段上传。服务器会将 ZIP 解压到本次请求的输入目录，并保留其中的子目录结构。

## 服务响应测试

```cmd
curl.exe "http://127.0.0.1:8510/health"
```

该接口只检查 FastAPI 服务是否能够正常响应，不会执行模型推理。从其他客户端访问时，需要将 `127.0.0.1` 替换为服务器的实际 IP 地址或域名。

`loaded_models` 用于查看当前进程已缓存的模型。例如服务刚启动时通常为：

```json
"loaded_models": [
  {"precision": "fp32", "device": "cuda"}
]
```

首次成功调用 `FP16batch_inference` 后，该数组会替换为：

```json
{"precision": "fp16", "device": "cuda"}
```

`loaded_models` 最多只有一个元素。

## 服务端进度日志

客户端调用 `/predict` 后，启动 Uvicorn 的服务端终端会实时显示请求和推理进度。日志包含 `request_id`、后端、设备、置信度、文件名、图片数量、单张图片检测结果、耗时和失败原因。

单图请求的日志示例：

```text
[3f1c...] Request accepted: backend=inference device=cuda conf=0.5000
[3f1c...] Image received: file=test.jpg size=245812 bytes resolution=1920x1080
[3f1c...] Inference started: backend=inference device=cuda conf=0.5000 text_prompt=yes box_prompts=0
[3f1c...] Inference completed: detections=3 inference_elapsed=1.284s total_elapsed=1.392s output_dir=...
```

批处理请求会输出每张图片的处理进度。例如：

```text
[82ab...] Batch backend starting: backend=batch_inference device=cuda precision=fp32 conf=0.5000 workers=1 model_cache=ready
[82ab...][batch_inference] image=camera_1/001.jpg detections=2 elapsed=1.102s
[82ab...] Batch request completed: completed=20 total=20 failed=0 detections=36 elapsed=24.531s output_dir=...
```

同一个请求的日志具有相同的 `request_id`，可据此跟踪一次 curl 调用的完整执行过程。日志仅输出到服务端终端，不会增加或修改接口返回 JSON 字段。

## 单图推理

Windows CMD：

```cmd
curl.exe -X POST "http://127.0.0.1:8510/predict" ^
  -F "image=@segmentation_input/test.jpg" ^
  -F "text=person" ^
  -F "backend=inference" ^
  -F "device=cuda" ^
  -F "conf=0.5"
```

以下所有多行 curl 示例均使用 Windows CMD 的 `^` 作为换行符。`^` 必须是该行最后一个字符，后面不能有空格。

## 批量推理

### 使用 `batch_inference` 后端

```cmd
curl.exe -X POST "http://127.0.0.1:8510/predict" ^
  -F "archive=@segmentation_input.zip" ^
  -F "text=person" ^
  -F "backend=batch_inference" ^
  -F "device=cuda" ^
  -F "conf=0.5"
```

### 使用 `FP16batch_inference` 后端

```cmd
curl.exe -X POST "http://127.0.0.1:8510/predict" ^
  -F "archive=@segmentation_input.zip" ^
  -F "text=person" ^
  -F "backend=FP16batch_inference" ^
  -F "device=cuda" ^
  -F "conf=0.5"
```

### 使用 `parallel_batch_inference` 后端

```cmd
curl.exe -X POST "http://127.0.0.1:8510/predict" ^
  -F "archive=@segmentation_input.zip" ^
  -F "text=person" ^
  -F "backend=parallel_batch_inference" ^
  -F "num_processes=4" ^
  -F "device=cuda" ^
  -F "conf=0.5"
```

在 PowerShell 或 Linux 中使用时，可以将整条 curl 指令写在同一行，避免不同终端的换行符差异。

## 响应 JSON 字段说明

当 `SAVE_RESULTS_LOCALLY = True` 时，上传文件和推理结果会保存到 `segmentation_output/api/{request_id}/`。`{request_id}` 使用服务端接收请求时的本地时间，格式为 `YYYY-MM-DD_HH-mm-ss-ffffff`，例如 `2026-08-17_14-32-08-123456`。当该配置为 `False` 时，响应内容不变，但请求目录会在响应返回前被删除。

### 公共顶层字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | string | 服务端接收请求时生成的时间标识，格式为 `YYYY-MM-DD_HH-mm-ss-ffffff`，同时也是本次请求输出目录的名称。 |
| `backend` | string | 实际处理请求的后端，可为 `inference`、`batch_inference`、`FP16batch_inference` 或 `parallel_batch_inference`。 |
| `device` | string | 客户端请求的标准化推理设备，例如 `cuda`、`cpu`、`npu` 或 `cann`。如果请求的 Provider 不可用，ONNX Runtime 仍可能使用配置的备用 Provider。 |
| `conf` | number | 实际使用的置信度阈值。只有分数严格大于该值的检测结果才会被保留。 |
| `num_processes` | integer \| null | `parallel_batch_inference` 使用的并发线程数；其他后端返回 `null`。字段名为兼容已有 curl 请求而保留。 |
| `detections` | integer | 检测框数量。单图模式表示该图片的检测框数量；批处理模式表示所有成功图片的检测框总数。 |

### 单图 `inference` 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `result` | object | 为上传图片保存的 LabelMe JSON 内容，详细结构见“LabelMe 结果字段”。 |

### 批处理响应字段

`batch_inference`、`FP16batch_inference` 和 `parallel_batch_inference` 在返回中还会增加以下字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `total_images` | integer | 上传的 ZIP 中发现的受支持图片总数。 |
| `completed_images` | integer | 成功生成结果 JSON 的图片数量。 |
| `failed_images` | string[] | 未能生成结果 JSON 的图片相对路径。全部成功时为空数组。 |
| `results` | object[] | 每张成功处理图片对应一个结果对象，不包含失败图片。 |

顶层 `results` 数组中的每个对象包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `results[].image` | string | 图片相对于 ZIP 根目录的路径，例如 `camera_1/001.jpg`。 |
| `results[].detections` | integer | 当前图片的检测框数量。 |
| `results[].result` | object | 当前图片对应的 LabelMe JSON 内容。 |

### LabelMe 结果字段

单图响应中的 `result` 和批处理响应中的每个 `results[].result` 使用相同的 LabelMe 结构：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `version` | string | 推理代码写入的 LabelMe 格式版本，当前为 `5.2.1`。 |
| `flags` | object | LabelMe 图片级标记，当前为空对象。 |
| `shapes` | object[] | 应用 `conf` 阈值后保留的检测结果；没有结果通过阈值时为空数组。 |
| `imagePath` | string | LabelMe JSON 中保存的原始图片文件名。 |
| `imageData` | string \| null | 内嵌图片数据。当前 API 不内嵌图片，因此该字段为 `null`。 |
| `imageHeight` | integer | 原始图片高度，单位为像素。 |
| `imageWidth` | integer | 原始图片宽度，单位为像素。 |

`shapes` 中的每个检测对象包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `shapes[].label` | string | 检测使用的文本提示词。单图仅使用框提示词时，默认标签为 `object`。 |
| `shapes[].points` | number[][] | 两个像素坐标，格式为 `[[x1, y1], [x2, y2]]`，分别表示矩形左上角和右下角。 |
| `shapes[].group_id` | integer \| null | 可选的 LabelMe 分组 ID，当前为 `null`。 |
| `shapes[].description` | string | 检测置信度，例如 `score: 0.9234`。 |
| `shapes[].shape_type` | string | LabelMe 图形类型，当前为 `rectangle`。 |
| `shapes[].flags` | object | LabelMe 图形级标记，当前为空对象。 |

### 错误响应字段

当后端名称无效、缺少上传文件、ZIP 无效或参数超出允许范围时，响应包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `detail` | string \| object[] | 便于阅读的错误信息。FastAPI 请求格式校验失败时，可能返回结构化校验信息数组。 |

## 创建批处理 ZIP

Windows PowerShell：

```powershell
Compress-Archive -Path .\segmentation_input\* `
  -DestinationPath .\segmentation_input.zip -Force
```

Linux：

```bash
cd segmentation_input
zip -r ../segmentation_input.zip .
```

不要在请求中传递类似 `input_dir=C:\images` 的客户端本地路径。项目部署后，该路径会被解释为服务器上的路径，而不是运行 curl 的客户端路径。

## 框提示词

框提示词只支持单图 `inference` 后端：

```cmd
curl.exe -X POST "http://127.0.0.1:8510/predict" ^
  -F "image=@segmentation_input/test.jpg" ^
  -F "text=person" ^
  -F "boxes=pos:5384,2352,397,785" ^
  -F "device=cuda" ^
  -F "conf=0.5"
```

## 关闭服务

在运行 Uvicorn 的终端中按 `Ctrl+C`。
