"""Interleaved re-timing of every model x runtime, so CPU-state drift hits YOLO26 and YOLO11 equally.

Uses the artifacts bench.py already exported. 3 rounds; each round visits every (model, runtime) in turn,
50 images after 15 warm-up runs. Reports the median over rounds of each round's median inference time.
Writes results/retime_pinned.jsonl (or retime_unpinned.jsonl with --unpinned). Accuracy is not re-measured (it does not depend on timing).
"""
import glob
import json
import os
import time

import sys

os.environ["YOLO_AUTOINSTALL"] = "False"
import psutil  # noqa: E402

# Hybrid Intel CPUs on Windows: in "Best power efficiency" mode the scheduler moves a process to slower cores a few
# seconds after start-up (see ecore_repro.py / README). Pinning to P-cores recovers most of it; Best performance mode
# avoids it. Pin unless --unpinned. i7-13650HX: logical CPUs 0-11 = 6 P-cores x 2 HT.
PINNED = "--unpinned" not in sys.argv
if PINNED:
    psutil.Process().cpu_affinity(list(range(12)))
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from bench import on_ac_power  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights")
IMGS = sorted(glob.glob(os.path.join(HERE, "data", "coco500", "images", "val", "*.jpg")))[:50]
MODELS = ["yolo26n", "yolo11n", "yolo26s", "yolo11s", "yolo26m", "yolo11m"]
RUNTIMES = {  # name -> (artifact suffix, device, half)
    "pytorch_cpu": (".pt", "cpu", False),
    "onnx_cpu": (".onnx", "cpu", False),
    "openvino_fp32_cpu": ("_openvino_model", "intel:cpu", False),
    "openvino_int8_cpu": ("_int8_openvino_model", "intel:cpu", False),
    "pytorch_cuda_fp32": (".pt", "cuda:0", False),
    "pytorch_cuda_fp16": (".pt", "cuda:0", True),
}
ROUNDS, WARM = 3, 15


def time_one(m, rt):
    suf, dev, half = RUNTIMES[rt]
    model = YOLO(os.path.join(W, m + suf), task="detect")
    for p in IMGS[:WARM]:
        model.predict(p, imgsz=640, device=dev, half=half, verbose=False)
    inf, wall = [], []
    for p in IMGS:
        t = time.perf_counter()
        r = model.predict(p, imgsz=640, device=dev, half=half, verbose=False)[0]
        wall.append((time.perf_counter() - t) * 1000)
        inf.append(r.speed["inference"])
    return float(np.median(inf)), float(np.median(wall))


if __name__ == "__main__":
    if not on_ac_power():
        raise SystemExit("On battery: plug in first.")
    acc = {(m, rt): [] for m in MODELS for rt in RUNTIMES}
    for rnd in range(ROUNDS):
        for rt in RUNTIMES:
            for m in MODELS:
                acc[(m, rt)].append(time_one(m, rt))
                print("round %d %-18s %-8s inf %.1f ms" % (rnd, rt, m, acc[(m, rt)][-1][0]), flush=True)
    with open(os.path.join(HERE, "results", "retime_pinned.jsonl" if PINNED else "retime_unpinned.jsonl"), "w") as f:
        for (m, rt), v in acc.items():
            inf = [x[0] for x in v]
            wall = [x[1] for x in v]
            f.write(json.dumps({"model": m, "config": rt, "inf_ms": round(float(np.median(inf)), 2),
                                "inf_rounds": [round(x, 2) for x in inf], "wall_ms": round(float(np.median(wall)), 2),
                                "fps_wall": round(1000 / float(np.median(wall)), 1), "ac_power": on_ac_power(), "pinned_pcores": PINNED}) + "\n")
