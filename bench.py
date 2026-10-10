"""YOLO26 vs YOLO11 on a laptop: CPU (PyTorch / ONNX / OpenVINO FP32 / OpenVINO INT8) and Intel iGPU (OpenVINO).

GPU: PyTorch CUDA FP32/FP16 on an RTX 4060 Laptop (TensorRT skipped: its export needs nvidia-modelopt).
Accuracy: mAP50-95 on a 500-image COCO val2017 subset. Latency: mean pre+inference+post per image,
batch 1, imgsz 640, 200 images after 10 warm-up runs. Results append to results/results.jsonl (resumable).
"""
import glob
import json
import os
import platform
import time

import numpy as np
os.environ["YOLO_AUTOINSTALL"] = "False"  # never let Ultralytics pip-install mid-benchmark
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "coco500.yaml")
IMGS = sorted(glob.glob(os.path.join(HERE, "data", "coco500", "images", "val", "*.jpg")))
OUT = os.path.join(HERE, "results", "results.jsonl")
WEIGHTS = os.path.join(HERE, "weights")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(WEIGHTS, exist_ok=True)

MODELS = {
    "yolo26n": "yolo26n.pt",  # copied from Ultralytics/YOLO26 on Hugging Face
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
}
# (name, export format, export kwargs, runtime device)
CONFIGS = [
    ("pytorch_cpu", None, {}, "cpu"),
    ("onnx_cpu", "onnx", {}, "cpu"),
    ("openvino_fp32_cpu", "openvino", {}, "intel:cpu"),
    ("openvino_int8_cpu", "openvino", {"int8": True, "data": DATA}, "intel:cpu"),
    ("pytorch_cuda_fp32", None, {}, "cuda:0"),
    ("pytorch_cuda_fp16", None, {"half": True}, "cuda:0"),
]


def done():
    if not os.path.exists(OUT):
        return set()
    return {(r["model"], r["config"]) for r in map(json.loads, open(OUT))}


def latency(model, device, half=False, n=200, warm=10):
    for p in IMGS[:warm]:
        model.predict(p, imgsz=640, device=device, half=half, verbose=False)
    pre, inf, post, wall = [], [], [], []
    for p in IMGS[:n]:
        t = time.perf_counter()
        r = model.predict(p, imgsz=640, device=device, half=half, verbose=False)[0]
        wall.append((time.perf_counter() - t) * 1000)
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
    return {"pre_ms": np.mean(pre), "inf_ms": np.mean(inf), "inf_p50_ms": np.median(inf), "inf_p90_ms": np.percentile(inf, 90),
            "post_ms": np.mean(post), "wall_ms": np.mean(wall), "fps_wall": 1000 / np.mean(wall)}


def main():
    os.chdir(WEIGHTS)  # exports and YOLO11 auto-downloads land here
    finished = done()
    for mname, src in MODELS.items():
        base = YOLO(src)
        for cname, fmt, kw, device in CONFIGS:
            if (mname, cname) in finished:
                continue
            row = {"model": mname, "config": cname, "device": device, "ac_power": on_ac_power()}
            half = False
            try:
                t = time.time()
                if fmt:
                    suffix = "_int8" if kw.get("int8") else ""
                    path = base.export(format=fmt, imgsz=640, batch=1, **kw)
                    model = YOLO(path, task="detect")
                    row["export_s"] = round(time.time() - t, 1)
                    row["size_mb"] = round(sum(os.path.getsize(f) for f in glob.glob(os.path.join(path, "*")) if os.path.isfile(f)) / 2**20
                                           if os.path.isdir(path) else os.path.getsize(path) / 2**20, 1)
                    row["artifact"] = os.path.basename(path) + suffix
                else:
                    model = YOLO(src)  # fresh copy: val() can leave the shared model on cuda / in fp16
                    half = kw.get("half", False)
                    row["size_mb"] = round(os.path.getsize(base.ckpt_path) / 2**20, 1)
                m = model.val(data=DATA, imgsz=640, batch=1, device=device, half=half, verbose=False, plots=False)
                row["map50_95"] = float(m.box.map)
                row["map50"] = float(m.box.map50)
                row.update({k: round(float(v), 2) for k, v in latency(model, device, half).items()})
            except Exception as e:  # keep going; record why a config failed
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            print(json.dumps(row), flush=True)
            with open(OUT, "a") as f:
                f.write(json.dumps(row) + "\n")


def on_ac_power():
    """Windows only: True when plugged in. Battery throttling ruins latency numbers."""
    import ctypes

    class SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte), ("BatteryLifePercent", ctypes.c_byte),
                    ("SystemStatusFlag", ctypes.c_byte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    s = SPS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))
    return s.ACLineStatus == 1


if __name__ == "__main__":
    print(platform.processor(), flush=True)
    if not on_ac_power():
        raise SystemExit("On battery: plug in and set Windows power mode to Best performance before benchmarking.")
    main()
