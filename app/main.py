"""离线采访素材多录音机时钟联合标定服务。

参考设备时钟直接作为统一时间；其余每台设备通过一个正斜率仿射映射
``unified = a * device + b`` 校准到统一时间。所有设备的成对同步点
联立为一个最小二乘问题统一求解。
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("clockcalib")

# --------------------------------------------------------------------------- #
# 请求 / 响应模型
# --------------------------------------------------------------------------- #


class DeviceInfo(BaseModel):
    id: str = Field(..., min_length=1, description="设备 ID")
    kind: Literal["recorder", "reference"] = Field(
        ..., description="reference 为参考设备，recorder 为待标定录音机"
    )


class SyncPoint(BaseModel):
    """一对同步点：两台设备在同一真实时刻各自的毫秒读数。"""

    model_config = ConfigDict(extra="forbid")

    device_a: str = Field(..., min_length=1)
    reading_a: float = Field(..., description="设备 A 在该同步点的毫秒读数")
    device_b: str = Field(..., min_length=1)
    reading_b: float = Field(..., description="设备 B 在该同步点的毫秒读数")


class Segment(BaseModel):
    """某台设备上的一个素材片段，起止为该设备本地毫秒读数。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="片段 ID")
    device: str = Field(..., min_length=1, description="片段所属设备 ID")
    start: float = Field(..., description="片段起点（设备本地毫秒）")
    end: float = Field(..., description="片段终点（设备本地毫秒）")


class CalibrateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    devices: list[DeviceInfo] = Field(..., min_length=1)
    reference_device: str = Field(..., min_length=1)
    sync_points: list[SyncPoint] = Field(..., min_length=1)
    residual_tolerance_ms: float = Field(
        ...,
        ge=0.0,
        description="非负残差容限（毫秒），任一同步点残差超出即拒绝",
    )
    segments: list[Segment] = Field(default_factory=list)


class Coefficients(BaseModel):
    slope: float
    intercept: float
    note: str | None = None


class UnifiedSegment(BaseModel):
    id: str
    device: str
    start: float = Field(..., description="统一时间下的起点（毫秒）")
    end: float = Field(..., description="统一时间下的终点（毫秒）")
    duration: float


class CalibrateResponse(BaseModel):
    reference_device: str
    coefficients: dict[str, Coefficients]
    segments: list[UnifiedSegment]


# --------------------------------------------------------------------------- #
# 标定核心
# --------------------------------------------------------------------------- #


def calibrate(req: CalibrateRequest) -> CalibrateResponse:
    device_ids = [d.id for d in req.devices]
    if len(device_ids) != len(set(device_ids)):
        raise CalibError("devices 中存在重复的设备 ID。")

    id_set = set(device_ids)
    if req.reference_device not in id_set:
        raise CalibError(
            f"参考设备 '{req.reference_device}' 未知：必须出现在 devices 列表中。"
        )

    ref_kind = {d.id: d.kind for d in req.devices}[req.reference_device]
    if ref_kind != "reference":
        raise CalibError(
            f"设备 '{req.reference_device}' 的 kind 为 {ref_kind!r}，"
            "参考设备必须声明为 kind=reference。"
        )
    if sum(d.kind == "reference" for d in req.devices) > 1:
        raise CalibError("reference 设备有且只能有一台。")

    # 非参考设备（固定顺序，决定未知量列位置）
    others = [d.id for d in req.devices if d.id != req.reference_device]
    col = {dev: i for i, dev in enumerate(others)}  # 每台设备两列：2i 斜率, 2i+1 截距
    n_unknown = 2 * len(others)

    def map_reading(dev: str, reading: float) -> tuple[int, float]:
        """返回 (矩阵列下标, 系数)，参考设备直接进入右端常数项。"""
        if not np.isfinite(reading):
            raise CalibError(f"设备 '{dev}' 的同步点读数必须是有限数值。")
        if dev == req.reference_device:
            return -1, reading
        if dev not in col:
            raise CalibError(f"同步点引用了未知设备 '{dev}'。")
        return col[dev] * 2, reading

    # 组装联立最小二乘  A x = y
    rows: list[tuple[int, float, int, float]] = []
    for sp in req.sync_points:
        if sp.device_a == sp.device_b:
            raise CalibError(
                f"同步点两端不能是同一台设备（'{sp.device_a}'）。"
            )
        ca, va = map_reading(sp.device_a, sp.reading_a)
        cb, vb = map_reading(sp.device_b, sp.reading_b)
        rows.append((ca, va, cb, vb))

    n_rows = len(rows)
    A = np.zeros((n_rows, n_unknown), dtype=np.float64)
    y = np.zeros(n_rows, dtype=np.float64)
    for r, (ca, va, cb, vb) in enumerate(rows):
        # 统一时刻相等：t(va) - t(vb) = 0
        # 设备端放左端（a 侧取正、b 侧取负），参考端移到右端故符号翻转
        if ca < 0:
            y[r] -= va
        else:
            A[r, ca] += va      # a_dev * reading
            A[r, ca + 1] += 1.0  # b_dev
        if cb < 0:
            y[r] += vb
        else:
            A[r, cb] -= vb
            A[r, cb + 1] -= 1.0

    # 方程不满列秩即拒绝：没有足够独立方程确定全部 (a, b)。
    # 按列范数归一化后判秩，消除“斜率列（×读数）与截距列（×1）尺度差异”的影响。
    col_norms = np.linalg.norm(A, axis=0)
    zero_cols = np.where(col_norms == 0)[0]
    scale = np.divide(
        1.0, col_norms, out=np.zeros_like(col_norms), where=col_norms != 0
    )
    try:
        rank = int(np.linalg.matrix_rank(A * scale))
    except np.linalg.LinAlgError as exc:  # 理论上不会发生，兜底
        raise CalibError(f"设计矩阵秩计算失败：{exc}") from exc
    if rank < n_unknown:
        missing = [
            dev for dev in others if col[dev] * 2 in zero_cols
        ] or [
            dev
            for dev in others
            if col[dev] * 2 + 1 in zero_cols
        ]
        detail = (
            f"，未参与任何同步点的设备：{', '.join(missing)}" if missing else ""
        )
        raise CalibError(
            f"方程组不满列秩（秩 {rank} / 未知量 {n_unknown}）："
            f"每台非参考设备都必须通过同步点与其它设备（最终连通到参考设备）相连{detail}。"
        )

    sol, residuals_raw, rank_lstsq, _ = np.linalg.lstsq(A, y, rcond=None)
    if rank_lstsq < n_unknown:
        # 数值层面的秩亏，matrix_rank 未捕获的极端情况
        raise CalibError("最小二乘求解出现数值秩亏，无法唯一确定各设备映射。")

    # 校验正斜率
    bad_slopes = [
        (dev, float(sol[col[dev] * 2]))
        for dev in others
        if sol[col[dev] * 2] <= 0
    ]
    if bad_slopes:
        desc = ", ".join(f"{dev} 的斜率 a={a:.6g}" for dev, a in bad_slopes)
        raise CalibError(f"检测到非正斜率，映射要求严格正斜率：{desc}。")

    # 残差校验（逐同步点，单位毫秒）
    residual = A @ sol - y
    over: list[str] = []
    for i, sp in enumerate(req.sync_points):
        res = float(abs(residual[i]))
        if res > req.residual_tolerance_ms:
            over.append(
                f"#{i + 1}({sp.device_a}={sp.reading_a:g} ↔ "
                f"{sp.device_b}={sp.reading_b:g}) 残差 {res:.6f}ms"
            )
    if over:
        raise CalibError(
            f"{len(over)} 个同步点残差超过容限 "
            f"{req.residual_tolerance_ms:g}ms：" + "；".join(over) + "。"
        )

    coeffs: dict[str, Coefficients] = {
        req.reference_device: Coefficients(
            slope=1.0, intercept=0.0, note="参考设备，映射固定为统一时间"
        )
    }
    for dev in others:
        a = float(sol[col[dev] * 2])
        b = float(sol[col[dev] * 2 + 1])
        coeffs[dev] = Coefficients(slope=a, intercept=b)

    # 片段映射：未知设备与起止颠倒在此报错
    unified_segments: list[UnifiedSegment] = []
    for seg in req.segments:
        if seg.device not in id_set:
            raise CalibError(
                f"片段 '{seg.id}' 引用了未知设备 '{seg.device}'。"
            )
        if not (np.isfinite(seg.start) and np.isfinite(seg.end)):
            raise CalibError(f"片段 '{seg.id}' 的起止时间必须是有限数值。")
        if seg.start > seg.end:
            raise CalibError(
                f"片段 '{seg.id}'（设备 '{seg.device}'）起止颠倒："
                f"start={seg.start:g} 晚于 end={seg.end:g}。"
            )
        c = coeffs[seg.device]
        u_start = c.slope * seg.start + c.intercept
        u_end = c.slope * seg.end + c.intercept
        unified_segments.append(
            UnifiedSegment(
                id=seg.id,
                device=seg.device,
                start=u_start,
                end=u_end,
                duration=u_end - u_start,
            )
        )

    unified_segments.sort(key=lambda s: (s.start, s.end, s.id))

    return CalibrateResponse(
        reference_device=req.reference_device,
        coefficients=coeffs,
        segments=unified_segments,
    )


class CalibError(Exception):
    """业务校验失败，映射为 HTTP 422。"""


# --------------------------------------------------------------------------- #
# FastAPI
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="多录音机时钟联合标定服务",
    description=(
        "离线采访素材多设备时钟标定：参考设备固定为原时间，"
        "其余设备用正斜率仿射映射，全部同步点联立最小二乘求解。"
    ),
    version="1.0.0",
)


@app.exception_handler(CalibError)
async def calib_error_handler(_request, exc: CalibError):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=422,
        content={"detail": [{"msg": str(exc), "type": "calibration_error"}]},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate_endpoint(req: CalibrateRequest) -> CalibrateResponse:
    # NumPy 计算放入线程池，避免阻塞事件循环
    return await run_in_threadpool(calibrate, req)
