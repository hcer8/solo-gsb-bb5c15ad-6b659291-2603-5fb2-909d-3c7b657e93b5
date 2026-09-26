import pytest
from fastapi.testclient import TestClient

from app.main import CalibrateRequest, CalibError, app, calibrate

client = TestClient(app)


def base_payload(**over):
    payload = {
        "devices": [
            {"id": "R0", "kind": "reference"},
            {"id": "A", "kind": "recorder"},
            {"id": "B", "kind": "recorder"},
        ],
        "reference_device": "R0",
        # A: t = 1.1 a + 5；B: t = 0.9 b - 2
        "sync_points": [
            {"device_a": "R0", "reading_a": 1105.0, "device_b": "A", "reading_b": 1000.0},
            {"device_a": "R0", "reading_a": 2205.0, "device_b": "A", "reading_b": 2000.0},
            {"device_a": "R0", "reading_a": 898.0, "device_b": "B", "reading_b": 1000.0},
            {"device_a": "R0", "reading_a": 1798.0, "device_b": "B", "reading_b": 2000.0},
            {"device_a": "A", "reading_a": 1000.0, "device_b": "B", "reading_b": 1230.0},
        ],
        "residual_tolerance_ms": 5.0,
        "segments": [
            {"id": "s2", "device": "B", "start": 1000.0, "end": 2000.0},
            {"id": "s1", "device": "A", "start": 1000.0, "end": 2000.0},
            {"id": "s0", "device": "R0", "start": 0.0, "end": 100.0},
        ],
    }
    payload.update(over)
    return payload


def test_basic_calibration_coefficients():
    resp = client.post("/calibrate", json=base_payload())
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["reference_device"] == "R0"
    assert data["coefficients"]["R0"] == {
        "slope": 1.0,
        "intercept": 0.0,
        "note": "参考设备，映射固定为统一时间",
    }
    a = data["coefficients"]["A"]
    b = data["coefficients"]["B"]
    assert a["slope"] == pytest.approx(1.1, abs=1e-9)
    assert a["intercept"] == pytest.approx(5.0, abs=1e-9)
    assert b["slope"] == pytest.approx(0.9, abs=1e-9)
    assert b["intercept"] == pytest.approx(-2.0, abs=1e-9)

    segs = data["segments"]
    # 按统一时间起点排序：s0=0, s2=0.9*1000-2=898, s1=1.1*1000+5=1105
    assert [s["id"] for s in segs] == ["s0", "s2", "s1"]
    assert segs[0] == pytest.approx(
        {"id": "s0", "device": "R0", "start": 0.0, "end": 100.0, "duration": 100.0},
        abs=1e-9,
    )
    assert segs[1]["id"] == "s2"
    assert segs[1]["start"] == pytest.approx(898.0, abs=1e-9)
    assert segs[1]["end"] == pytest.approx(1798.0, abs=1e-9)
    assert segs[2]["id"] == "s1"
    assert segs[2]["start"] == pytest.approx(1105.0, abs=1e-9)
    assert segs[2]["end"] == pytest.approx(2205.0, abs=1e-9)


def test_chain_without_direct_reference_points():
    """B 只与 A 同步，不直接连参考设备，也应能联合解出。"""
    payload = base_payload(
        sync_points=[
            {"device_a": "R0", "reading_a": 1105.0, "device_b": "A", "reading_b": 1000.0},
            {"device_a": "R0", "reading_a": 2205.0, "device_b": "A", "reading_b": 2000.0},
            {"device_a": "A", "reading_a": 1105.0, "device_b": "B", "reading_b": 1000.0},
            {"device_a": "A", "reading_a": 2205.0, "device_b": "B", "reading_b": 2000.0},
        ],
        segments=[],
    )
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 200, resp.text
    b = resp.json()["coefficients"]["B"]
    # t_B(1000)=t_A(1105)=1220.5, t_B(2000)=t_A(2205)=2430.5
    assert b["slope"] == pytest.approx(1.21, abs=1e-9)
    assert b["intercept"] == pytest.approx(10.5, abs=1e-9)


def test_rank_deficient_is_rejected():
    payload = base_payload(
        sync_points=[
            {"device_a": "R0", "reading_a": 1105.0, "device_b": "A", "reading_b": 1000.0},
            {"device_a": "R0", "reading_a": 2205.0, "device_b": "A", "reading_b": 2000.0},
            # B 完全没有同步点
        ],
        segments=[],
    )
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 422
    msg = resp.json()["detail"][0]["msg"]
    assert "不满列秩" in msg and "B" in msg


def test_single_point_per_device_is_rank_deficient():
    payload = base_payload(
        sync_points=[
            {"device_a": "R0", "reading_a": 1105.0, "device_b": "A", "reading_b": 1000.0},
            {"device_a": "R0", "reading_a": 898.0, "device_b": "B", "reading_b": 1000.0},
        ],
        segments=[],
    )
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 422
    assert "不满列秩" in resp.json()["detail"][0]["msg"]


def test_non_positive_slope_rejected():
    payload = base_payload(
        sync_points=[
            # A 的时钟倒走：读数越大统一时间越小
            {"device_a": "R0", "reading_a": 2000.0, "device_b": "A", "reading_b": 1000.0},
            {"device_a": "R0", "reading_a": 1000.0, "device_b": "A", "reading_b": 2000.0},
            # B 正常，保证秩亏不是首要原因
            {"device_a": "R0", "reading_a": 898.0, "device_b": "B", "reading_b": 1000.0},
            {"device_a": "R0", "reading_a": 1798.0, "device_b": "B", "reading_b": 2000.0},
        ],
        segments=[],
    )
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 422
    msg = resp.json()["detail"][0]["msg"]
    assert "非正斜率" in msg and "A" in msg


def test_residual_over_tolerance_rejected():
    points = base_payload()["sync_points"]
    points[0]["reading_a"] = 1300.0  # 制造约 195ms 残差
    payload = base_payload(sync_points=points)
    resp = client.post("/calibrate", json=payload)
    assert resp.status_code == 422
    msg = resp.json()["detail"][0]["msg"]
    assert "残差超过容限" in msg and "5" in msg


def test_unknown_device_in_sync_point():
    points = base_payload()["sync_points"]
    points[0]["device_b"] = "GHOST"
    resp = client.post("/calibrate", json=base_payload(sync_points=points))
    assert resp.status_code == 422
    assert "未知设备" in resp.json()["detail"][0]["msg"]


def test_unknown_device_in_segment():
    resp = client.post(
        "/calibrate",
        json=base_payload(
            segments=[{"id": "x", "device": "GHOST", "start": 0, "end": 10}]
        ),
    )
    assert resp.status_code == 422
    assert "未知设备" in resp.json()["detail"][0]["msg"]


def test_unknown_reference_device():
    resp = client.post("/calibrate", json=base_payload(reference_device="NOPE"))
    assert resp.status_code == 422
    assert "参考设备" in resp.json()["detail"][0]["msg"]


def test_reversed_segment_rejected():
    resp = client.post(
        "/calibrate",
        json=base_payload(
            segments=[{"id": "x", "device": "A", "start": 200, "end": 100}]
        ),
    )
    assert resp.status_code == 422
    assert "起止颠倒" in resp.json()["detail"][0]["msg"]


def test_negative_tolerance_rejected_by_schema():
    resp = client.post("/calibrate", json=base_payload(residual_tolerance_ms=-1))
    assert resp.status_code == 422  # pydantic 422，非业务 422
    detail = resp.json()["detail"]
    assert isinstance(detail, list)
    assert all("msg" in d for d in detail)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_zero_length_segment_allowed():
    resp = client.post(
        "/calibrate",
        json=base_payload(
            segments=[{"id": "z", "device": "A", "start": 100, "end": 100}]
        ),
    )
    assert resp.status_code == 200, resp.text
    seg = resp.json()["segments"][-1]
    assert seg["duration"] == pytest.approx(0.0, abs=1e-12)


def test_core_function_directly():
    req = CalibrateRequest.model_validate(base_payload())
    result = calibrate(req)
    assert result.coefficients["A"].slope > 0
