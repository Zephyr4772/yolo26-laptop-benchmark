"""First-pass tables from results/results.jsonl. NOTE: its latency columns are sequential and E-core-affected;
the README latency table comes from results/retime_pinned.jsonl (interleaved, P-core pinned). Accuracy here is valid."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "results", "results.jsonl"))]
R = {(r["model"], r["config"]): r for r in rows}
SIZES = ["n", "s", "m"]
CFG = ["pytorch_cpu", "onnx_cpu", "openvino_fp32_cpu", "openvino_int8_cpu", "pytorch_cuda_fp32", "pytorch_cuda_fp16"]
LABEL = {"pytorch_cpu": "PyTorch CPU", "onnx_cpu": "ONNX CPU", "openvino_fp32_cpu": "OpenVINO FP32 CPU",
         "openvino_int8_cpu": "OpenVINO INT8 CPU", "pytorch_cuda_fp32": "PyTorch CUDA FP32",
         "pytorch_cuda_fp16": "PyTorch CUDA FP16", "tensorrt_fp16_cuda": "TensorRT FP16"}
# Official model-card numbers (CPU ONNX on EC2 P4d Xeon, T4 TensorRT10 FP16), YOLO26 only
CARD = {"n": (38.9, 1.7, 40.9), "s": (87.2, 2.5, 48.6), "m": (220.0, 4.7, 53.1)}


def g(m, c, k):
    r = R.get((m, c))
    return None if r is None or "error" in r else r.get(k)


def f(x, fmt="%.1f"):
    return "—" if x is None else fmt % x


print("### Errors")
for r in rows:
    if "error" in r:
        print("- %s / %s: %s" % (r["model"], r["config"], r["error"]))

print("\n### Inference latency (ms/img, mean, batch 1, 640) — YOLO26 vs YOLO11")
print("| runtime | " + " | ".join("26%s | 11%s | 26 vs 11" % (s, s) for s in SIZES) + " |")
print("|---|" + "---|---|---|" * len(SIZES))
for c in CFG:
    cells = []
    for s in SIZES:
        a, b = g("yolo26" + s, c, "inf_ms"), g("yolo11" + s, c, "inf_ms")
        cells += [f(a), f(b), "—" if not (a and b) else "%+.0f%%" % (100 * (a - b) / b)]
    print("| %s | %s |" % (LABEL[c], " | ".join(cells)))

print("\n### End-to-end FPS (pre + inference + post, wall clock)")
print("| runtime | " + " | ".join("26%s | 11%s" % (s, s) for s in SIZES) + " |")
print("|---|" + "---|---|" * len(SIZES))
for c in CFG:
    print("| %s | %s |" % (LABEL[c], " | ".join("%s | %s" % (f(g("yolo26" + s, c, "fps_wall"), "%.0f"), f(g("yolo11" + s, c, "fps_wall"), "%.0f")) for s in SIZES)))

print("\n### Accuracy (mAP50-95 on 500 COCO val images)")
print("| runtime | " + " | ".join("26%s | 11%s" % (s, s) for s in SIZES) + " |")
print("|---|" + "---|---|" * len(SIZES))
for c in CFG:
    print("| %s | %s |" % (LABEL[c], " | ".join("%s | %s" % (f(g("yolo26" + s, c, "map50_95"), "%.3f"), f(g("yolo11" + s, c, "map50_95"), "%.3f")) for s in SIZES)))

print("\n### Laptop vs model card (YOLO26)")
print("| size | card CPU ONNX (Xeon) | laptop ONNX CPU | laptop OpenVINO INT8 CPU | card T4 TensorRT FP16 | laptop PyTorch FP16 (RTX 4060) |")
print("|---|---|---|---|---|---|")
for s in SIZES:
    c = CARD[s]
    print("| %s | %.1f | %s | %s | %.1f | %s |" % (s, c[0], f(g("yolo26" + s, "onnx_cpu", "inf_ms")), f(g("yolo26" + s, "openvino_int8_cpu", "inf_ms")),
                                               c[1], f(g("yolo26" + s, "pytorch_cuda_fp16", "inf_ms"))))

print("\n### Where end-to-end time goes, fastest configs (ms)")
print("| model / runtime | pre | inference | post | wall |")
print("|---|---|---|---|---|")
for m in ["yolo26n", "yolo11n"]:
    for c in ["openvino_int8_cpu", "pytorch_cuda_fp16"]:
        print("| %s %s | %s | %s | %s | %s |" % (m, LABEL[c], f(g(m, c, "pre_ms")), f(g(m, c, "inf_ms")), f(g(m, c, "post_ms")), f(g(m, c, "wall_ms"))))
print("\nAll rows on AC power:", all(r.get("ac_power") for r in rows))
