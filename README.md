# YOLO26 vs YOLO11 on a gaming laptop: same speed, better nano accuracy, and Windows can cost you 2–3.5×

**Hardware:** i7-13650HX (6 P-cores + 8 E-cores) and an RTX 4060 Laptop GPU, Windows 11, on AC power.

**Software:** Ultralytics 8.4.78, ONNX Runtime 1.27, OpenVINO 2026.4, torch 2.5.1+cu121.

**What was tested:**
- **Models:** YOLO26 n/s/m against YOLO11 n/s/m.
- **6 runtimes:**
  - **CPU:** PyTorch, ONNX, OpenVINO FP32, OpenVINO INT8.
  - **GPU:** PyTorch CUDA FP32 and FP16.
- **Settings:** batch 1, 640 px.
- **Accuracy data:** 500 COCO val2017 images.

## TL;DR

1. **On the same machine, measured fairly, YOLO26 and YOLO11 run at about the same speed.** They're within ±15% on every CPU runtime, and YOLO11 is about 15% faster on the GPU at batch 1.
2. **YOLO26's gain is accuracy, mostly at nano size:** +1.7 to +2.1 mAP for n (+1.0 after INT8), and +0.3 to +0.4 for s and m (that gap disappears after INT8).
3. **The runtime matters far more than the model version.** OpenVINO INT8 is **3.6–4.2× faster** than PyTorch on the CPU, at a cost of about 0.5–1.6 mAP.
4. **Watch out for Windows on hybrid Intel CPUs.** Once `torchvision` is imported (Ultralytics does this), Windows tends to move the process onto E-cores, which made every runtime **1.7–3.5× slower**. Pinning to P-cores fixes it.
5. **On the GPU at batch 1, n, s and m all take about 10–12 ms.** At this size the time goes to launch overhead, not compute, and FP16 gains nothing.

## Inference latency, pinned to P-cores

The figures are ms per image, inference only, as the median of 3 interleaved rounds × 50 images.

| runtime | 26n | 11n | 26s | 11s | 26m | 11m |
|---|---|---|---|---|---|---|
| PyTorch CPU | 52.1 | 48.4 | 93.3 | 88.0 | 199.2 | 196.2 |
| ONNX CPU | 37.6 | 44.2 | 102.6 | 95.2 | 226.9 | 218.0 |
| OpenVINO FP32 CPU | 25.2 | 28.9 | 60.0 | 60.4 | 176.9 | 175.7 |
| **OpenVINO INT8 CPU** | **13.2** | 14.6 | **22.4** | 21.8 | **55.2** | 55.8 |
| PyTorch CUDA FP32 | 10.3 | 8.8 | 10.3 | 8.8 | 12.7 | 11.1 |
| PyTorch CUDA FP16 | 10.9 | 9.4 | 11.2 | 10.1 | 12.1 | 12.1 |

**End to end** (pre-processing + inference + post-processing), YOLO26n reaches about **57 FPS on OpenVINO INT8 (CPU)** and about **74 FPS on CUDA**.

## Accuracy (mAP50-95, 500 COCO val images)

| runtime | 26n | 11n | 26s | 11s | 26m | 11m |
|---|---|---|---|---|---|---|
| PyTorch | 43.2 | 41.5 | 50.1 | 49.8 | 55.0 | 54.6 |
| ONNX / OpenVINO FP32 | 44.2 | 42.1 | 50.0 | 49.6 | 55.4 | 55.0 |
| OpenVINO INT8 | 42.6 | 41.6 | 48.8 | 48.9 | 54.3 | 54.4 |

- **These are subset numbers.** The model card's full-val figures (26n 40.9, 26s 48.6, 26m 53.1) are lower; 500 images is not the full set.
- **PyTorch and the exported runtimes differ by up to 1 mAP.** Exported models use a fixed 640×640 letterbox, while PyTorch uses tighter padding.
- **INT8 was calibrated on these same 500 images,** so its accuracy is slightly optimistic.

## The Windows E-core trap

| runtime (YOLO26n) | unpinned ÷ pinned latency |
|---|---|
| PyTorch CPU | 2.4× |
| ONNX CPU | 3.5× |
| OpenVINO FP32 | 1.9× |
| OpenVINO INT8 | 1.7× |
| CUDA FP32 / FP16 | 1.9× |

**Minimal repro:** a pure-Python loop takes about 98 ms. After `import torchvision`, the same loop takes 200–260 ms. If you first pin the process to P-cores, it takes 109 ms.

**Fix:** pin to P-cores (`psutil.Process().cpu_affinity(list(range(12)))` on this CPU), or use Windows *Best performance* mode. Otherwise every CPU benchmark on these laptops is a coin flip, and so is your production FPS.

## How I almost published the wrong headline

**My first sequential run showed YOLO26 50–90% faster than YOLO11 on the CPU.** It was false. YOLO26 n/s happened to run while the process sat on P-cores, and YOLO11 ran after Windows had moved it to E-cores.

**Re-timing every model × runtime in interleaved rounds removed the gap.** If you benchmark on a hybrid laptop, interleave the runs, pin the cores, and stay plugged in. A battery run of the same models was 1.3–4× slower again; it is discarded here.

## Laptop vs the model card (YOLO26)

| | n | s | m |
|---|---|---|---|
| card, ONNX CPU (EC2 Xeon) | 38.9 | 87.2 | 220.0 |
| this laptop, ONNX CPU, pinned | 37.6 | 102.6 | 226.9 |
| this laptop, OpenVINO INT8 | 13.2 | 22.4 | 55.2 |
| card, TensorRT FP16 (T4) | 1.7 | 2.5 | 4.7 |
| this laptop, PyTorch CUDA FP16 | 10.9 | 11.2 | 12.1 |

- **The ONNX CPU numbers track the card closely.**
- **The GPU gap is the runtime, not the hardware.** The card uses TensorRT, while PyTorch eager mode at batch 1 is overhead-bound. TensorRT was not tested here; see the caveats.

## Caveats

- **Accuracy is on a subset:** 500 images, not the full COCO val set.
- **Single laptop, single run of the accuracy pass.** Latency is the median of 3 interleaved rounds; the worst round-to-round spread was 1.6×.
- **TensorRT was not tested.** The Ultralytics 8.4.78 TensorRT export auto-installs `nvidia-modelopt`. On this Windows Python, that install tried to replace torch and half-removed `onnx`. I disabled auto-install (`YOLO_AUTOINSTALL=False`) and skipped TensorRT.
- **The INT8 calibration set overlaps the evaluation set.**
- **P-core indices 0–11 are specific to the i7-13650HX.** Check your own CPU's layout before pinning.

## Reproduce

```bash
pip install -r requirements.txt
# data: COCO val2017 labels (Ultralytics coco2017labels.zip) + the 500 images listed in data_list/pick.txt from images.cocodataset.org
py -3.12 bench.py      # exports + accuracy + first-pass latency -> results/results.jsonl
py -3.12 retime.py     # interleaved, P-core pinned latency -> results/retime_pinned.jsonl
py -3.12 retime.py --unpinned
py -3.12 analyze.py
```

YOLO26 weights come from [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26). YOLO11 weights are auto-downloaded by Ultralytics.
