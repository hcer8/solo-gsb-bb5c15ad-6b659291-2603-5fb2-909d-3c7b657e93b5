# 离线采访素材 · 多录音机时钟联合标定服务

离线采访（多台录音机分别录制）场景下，用成对的同步点（如对拍声、打板声）把
所有设备的本地毫秒时钟联合标定到统一时间轴的 Python FastAPI 服务。

- **参考设备**：时钟固定为统一时间，映射 `t = r`（斜率 1、截距 0）。
- **其余每台设备**：一个**正斜率**仿射映射 `t = slope × r + intercept`。
- **求解方式**：把所有成对同步点联立为一个线性最小二乘问题，用 NumPy 一次解出
  全部设备的系数；随后逐点检查残差。
- 标定成功后，请求中的每个素材片段（起止为设备本地毫秒）被映射到统一时间，
  并按统一时间起点升序返回。

任一下列情况会拒绝请求（HTTP 422）并在返回体 `detail[0].msg` 中说明原因：

1. 方程组**不满列秩**（有设备未被任何同步点约束，或同步网络最终连不到参考设备）；
2. 某台设备拟合出的**斜率非正**（时钟必须单调前进）；
3. **任一同步点残差超过**给定的非负残差容限；
4. 同步点或片段引用了**未知设备**；
5. 片段**起止颠倒**（`start > end`，等长片段 `start == end` 允许）；
6. 参考设备未知 / 未声明为 `reference`、重复设备 ID、容限为负等。

## 单条启动命令

需要本机已安装 Docker（含 Compose v2，即 `docker compose` 子命令）：

```bash
docker compose up --build
```

启动后服务监听在 **http://0.0.0.0:8000**：

| 用途 | 地址 |
| --- | --- |
| 健康检查 | http://localhost:8000/health |
| 标定接口（POST JSON） | http://localhost:8000/calibrate |
| 交互式 API 文档（Swagger UI） | http://localhost:8000/docs |
| OpenAPI 规范 | http://localhost:8000/openapi.json |

## 必要配置

均为可选项，不配也能直接用默认值启动：

| 配置方式 | 变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| 宿主机端口（端口映射左侧） | `HOST_PORT`（compose 变量，可用环境变量或项目根目录 `.env`） | `8000` | 例如 `HOST_PORT=9000 docker compose up --build` |
| 容器内监听地址 | `UVICORN_HOST` | `0.0.0.0` | 一般无需修改 |
| 容器内监听端口 | `UVICORN_PORT` | `8000` | 修改后需同步调整 compose 的端口映射右侧 |

依赖（NumPy / FastAPI / Uvicorn / Pydantic）在镜像构建期安装，容器启动即完成
初始化并对外提供服务，无需手工初始化步骤。

## 请求格式

`POST /calibrate`，`Content-Type: application/json`：

```json
{
  "devices": [
    {"id": "R0", "kind": "reference"},
    {"id": "A",  "kind": "recorder"},
    {"id": "B",  "kind": "recorder"}
  ],
  "reference_device": "R0",
  "sync_points": [
    {"device_a": "R0", "reading_a": 1105.0, "device_b": "A", "reading_b": 1000.0},
    {"device_a": "R0", "reading_a": 2205.0, "device_b": "A", "reading_b": 2000.0},
    {"device_a": "A",  "reading_a": 1000.0, "device_b": "B", "reading_b": 1230.0}
  ],
  "residual_tolerance_ms": 5.0,
  "segments": [
    {"id": "clip-1", "device": "A", "start": 1000.0, "end": 2000.0},
    {"id": "clip-2", "device": "B", "start": 1000.0, "end": 1500.0}
  ]
}
```

字段说明：

- `devices`：全部设备。`kind` 为 `reference`（有且仅一台）或 `recorder`。
- `reference_device`：参考设备 ID，必须在 `devices` 中。
- `sync_points`：成对同步点，含义为“同一真实时刻两台设备各自的毫秒读数”；
  两端可以是任意设备（录音机之间、录音机与参考设备之间均可），
  所有点联立参与同一次最小二乘。
- `residual_tolerance_ms`：非负残差容限（毫秒）。
- `segments`：可选，素材片段清单，`start`/`end` 为该设备本地毫秒。

## 返回示例

```json
{
  "reference_device": "R0",
  "coefficients": {
    "R0": {"slope": 1.0, "intercept": 0.0, "note": "参考设备，映射固定为统一时间"},
    "A":  {"slope": 1.1, "intercept": 5.0, "note": null},
    "B":  {"slope": 0.9, "intercept": -2.0, "note": null}
  },
  "segments": [
    {"id": "clip-2", "device": "B", "start": 898.0,  "end": 1348.0, "duration": 450.0},
    {"id": "clip-1", "device": "A", "start": 1105.0, "end": 2205.0, "duration": 1100.0}
  ]
}
```

`segments` 已按统一时间 `start` 升序排列。

快速试一条请求（服务已启动时）：

```bash
curl -X POST http://localhost:8000/calibrate \
  -H 'Content-Type: application/json' \
  --data @examples/request.json
```

## 本地开发（不使用 Docker）

Python 3.11+：

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
pytest -q          # 运行测试（需 pytest、httpx）
```

## 项目结构

```
app/main.py            # FastAPI 应用与 NumPy 联合标定逻辑
tests/test_api.py      # 接口与标定逻辑测试
Dockerfile             # 依赖在构建期安装，启动直接对外服务
docker-compose.yml     # 单条命令启动（端口、健康检查、重启策略）
requirements.txt
examples/request.json  # 请求示例
```
