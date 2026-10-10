"""TensorRT latency without the Ultralytics exporter (which auto-installs nvidia-modelopt and broke this env).

Builds an engine from the existing static ONNX export with the installed `tensorrt` package (FP16 if the
builder supports the flag, otherwise FP32/TF32) and times execute_async_v3 on real, preprocessed COCO images.
Inference only: host->device copy and sync included, pre/post-processing excluded. Interleaved rounds.
Writes results/tensorrt.jsonl.
"""
import glob
import json
import os
import time

import cv2
import numpy as np
import tensorrt as trt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights")
MODELS = ["yolo26n", "yolo11n", "yolo26s", "yolo11s", "yolo26m", "yolo11m"]
IMGS = sorted(glob.glob(os.path.join(HERE, "data", "coco500", "images", "val", "*.jpg")))[:100]
LOG = trt.Logger(trt.Logger.WARNING)


def letterbox(p):
    im = cv2.imread(p)
    h, w = im.shape[:2]
    r = 640 / max(h, w)
    im = cv2.resize(im, (round(w * r), round(h * r)))
    c = np.full((640, 640, 3), 114, np.uint8)
    top, left = (640 - im.shape[0]) // 2, (640 - im.shape[1]) // 2
    c[top:top + im.shape[0], left:left + im.shape[1]] = im
    return np.ascontiguousarray(c[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255)


def fp16_onnx(m):
    """TensorRT 11 has no FP16 builder flag: precision comes from the ONNX types, so export an FP16 ONNX."""
    dst = os.path.join(W, m + "_fp16.onnx")
    if not os.path.exists(dst):
        os.environ["YOLO_AUTOINSTALL"] = "False"
        from ultralytics import YOLO
        fp32 = os.path.join(W, m + ".onnx")
        backup = fp32 + ".fp32bak"
        os.replace(fp32, backup)  # the FP16 export writes to the same filename
        try:
            src = YOLO(os.path.join(W, m + ".pt")).export(format="onnx", imgsz=640, batch=1, half=True, device=0)
            os.replace(src, dst)
        finally:
            os.replace(backup, fp32)
    return dst


def build(m, fp16=False):
    path = os.path.join(W, m + ("_trt_fp16.engine" if fp16 else "_trt.engine"))
    precision = "fp16"  # becomes "fp32" below if the FP16 builder flag no longer exists
    if not os.path.exists(path):
        builder = trt.Builder(LOG)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if fp16 else 0
        net = builder.create_network(flags)
        parser = trt.OnnxParser(net, LOG)
        with open(fp16_onnx(m) if fp16 else os.path.join(W, m + ".onnx"), "rb") as f:
            if not parser.parse(f.read()):
                raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        if not fp16:
            try:
                cfg.set_flag(trt.BuilderFlag.FP16)
            except AttributeError:  # flag removed in TensorRT 11; FP32 engine from the FP32 ONNX
                precision = "fp32"
        blob = builder.build_serialized_network(net, cfg)
        if blob is None:
            raise RuntimeError("engine build failed")
        with open(path, "wb") as f:
            f.write(blob)
        with open(path + ".precision", "w") as f:
            f.write(precision)
    else:
        precision = open(path + ".precision").read().strip()
    eng = trt.Runtime(LOG).deserialize_cuda_engine(open(path, "rb").read())
    return eng, precision


def runner(eng):
    ctx = eng.create_execution_context()
    bufs = {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shape = tuple(eng.get_tensor_shape(n))
        dt = torch.float16 if eng.get_tensor_dtype(n) == trt.DataType.HALF else torch.float32
        bufs[n] = torch.empty(shape, dtype=dt, device="cuda")
        ctx.set_tensor_address(n, bufs[n].data_ptr())
    inp = [n for n in bufs if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    stream = torch.cuda.Stream()

    def run(x):
        with torch.cuda.stream(stream):
            bufs[inp].copy_(torch.from_numpy(x).to(bufs[inp].dtype), non_blocking=False)
            ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
    return run


if __name__ == "__main__":
    X = [letterbox(p) for p in IMGS]
    out, R = {}, {}
    for m0 in MODELS:
        for fp16 in (False, True):
            m = m0 + ("@fp16" if fp16 else "")
            try:
                eng, prec = build(m0, fp16)
                R[m] = (runner(eng), prec)
                print("built", m, prec, flush=True)
            except Exception as e:
                out[m] = {"model": m, "error": "%s: %s" % (type(e).__name__, str(e)[:300])}
                print(out[m], flush=True)
    times = {m: [] for m in R}
    for rnd in range(3):
        for m, (run, _) in R.items():
            for x in X[:20]:
                run(x)
            t = []
            for x in X:
                a = time.perf_counter()
                run(x)
                t.append((time.perf_counter() - a) * 1000)
            times[m].append(float(np.median(t)))
    for m, (run, prec) in R.items():
        out[m] = {"model": m.split("@")[0], "config": "tensorrt_" + prec, "trt_version": trt.__version__,
                  "inf_ms": round(float(np.median(times[m])), 3), "inf_rounds": [round(v, 3) for v in times[m]]}
    with open(os.path.join(HERE, "results", "tensorrt.jsonl"), "w") as f:
        for m in [x + y for x in MODELS for y in ("", "@fp16")]:
            if m in out:
                f.write(json.dumps(out[m]) + "\n")
                print(json.dumps(out[m]))
