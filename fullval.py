"""Accuracy on proper data (no latency here).

1. Full COCO val2017 (5,000 images), PyTorch on CUDA FP32, batch 32 -> comparable to the model card's mAP.
2. INT8 vs FP32 OpenVINO on the 4,500 val images NOT used for INT8 calibration (calibration = data_list/pick.txt).
Writes results/fullval.jsonl (resumable).
"""
import json
import os
import sys

os.environ["YOLO_AUTOINSTALL"] = "False"
import psutil  # noqa: E402

from ultralytics import YOLO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights")
OUT = os.path.join(HERE, "results", "fullval.jsonl")
MODELS = ["yolo26n", "yolo11n", "yolo26s", "yolo11s", "yolo26m", "yolo11m"]
JOBS = [  # (job, artifact suffix, data yaml, device, batch)
    ("full5000_cuda_fp32", ".pt", "coco_full.yaml", "cuda:0", 32),
    ("holdout4500_openvino_fp32", "_openvino_model", "coco_holdout4500.yaml", "intel:cpu", 1),
    ("holdout4500_openvino_int8", "_int8_openvino_model", "coco_holdout4500.yaml", "intel:cpu", 1),
]


def main():
    psutil.Process().cpu_affinity(list(range(12)))  # P-cores only (i7-13650HX); see README
    done = {(r["model"], r["job"]) for r in map(json.loads, open(OUT)) if "error" not in r} if os.path.exists(OUT) else set()
    for job, suf, data, dev, bs in JOBS:
        for m in MODELS:
            if (m, job) in done:
                continue
            row = {"model": m, "job": job}
            try:
                r = YOLO(os.path.join(W, m + suf), task="detect").val(data=os.path.join(HERE, data), imgsz=640, batch=bs,
                                                                       device=dev, verbose=False, plots=False)
                row.update(map50_95=round(float(r.box.map), 4), map50=round(float(r.box.map50), 4))
            except Exception as e:
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            print(json.dumps(row), flush=True)
            with open(OUT, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":  # required on Windows: val() spawns DataLoader worker processes
    main()
