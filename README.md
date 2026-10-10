# YOLO26 vs YOLO11 on a gaming laptop: same speed, +0.6–0.8 mAP, and Windows power-saving can cost you up to 8×

An independent benchmark of [Ultralytics YOLO26](https://huggingface.co/Ultralytics/YOLO26) against YOLO11 (n/s/m) on an ordinary gaming laptop.

**Hardware:** i7-13650HX (6 P-cores + 8 E-cores) and an RTX 4060 Laptop GPU, Windows 11, on AC power.

**Software:** Ultralytics 8.4.78, ONNX Runtime 1.27, OpenVINO 2026.4, TensorRT 11.4, torch 2.5.1+cu121.

**Runtimes tested:**
- **CPU:** PyTorch, ONNX, OpenVINO FP32, OpenVINO INT8.
- **GPU:** PyTorch CUDA FP32/FP16 and TensorRT FP32/FP16.

**Settings:** batch 1, 640 px.

## TL;DR

1. **Speed: a tie.** On the same laptop, YOLO26 and YOLO11 are within ±25% on every runtime, and mostly within ±5%. YOLO26n is about 20% faster on ONNX and OpenVINO FP32. On the GPU, YOLO11 is equal or up to about 20% faster (PyTorch FP16, TensorRT FP16 s/m).
2. **Accuracy: YOLO26 wins by a little.** On the full COCO val2017 set (5,000 images), YOLO26 leads by +0.6 (n), +0.8 (s) and +0.8 (m) mAP50-95. The lead survives INT8.
3. **The runtime matters far more than the model version.** On the CPU, OpenVINO INT8 runs YOLO26n at 9.8 ms (70 FPS end to end), against 52.7 ms in PyTorch, for about 1 mAP.
4. **Windows' power mode is the biggest factor of all.** In *Best power efficiency* mode (which Energy Saver can force), Windows moves a Python process to slower cores a few seconds after it starts. Every runtime got 2–8× slower. *Best performance* mode removes it.

## Accuracy (mAP50-95)

**Full COCO val2017, 5,000 images** (PyTorch, CUDA FP32, batch 32, end-to-end/NMS-free outputs):

| | n | s | m |
|---|---|---|---|
| YOLO26 | **39.5** | **47.2** | **51.8** |
| YOLO11 | 38.9 | 46.4 | 51.0 |
| YOLO26 lead | +0.6 | +0.8 | +0.8 |
| model card, YOLO26 (e2e) | 40.1 | 47.8 | 52.5 |

**These reproduce the card to within 0.6–0.7 mAP.** The gap is consistent with the evaluation settings (batch-32 rectangular padding, NMS-free mode).

**INT8 cost, held out:** OpenVINO INT8 was calibrated on 500 val images and is scored here on the other 4,500, which it never saw.

| | 26n | 11n | 26s | 11s | 26m | 11m |
|---|---|---|---|---|---|---|
| OpenVINO FP32 | 39.6 | 38.6 | 47.1 | 45.9 | 51.8 | 50.6 |
| OpenVINO INT8 | 38.7 | 38.1 | 45.7 | 45.2 | 50.5 | 49.8 |
| INT8 cost | −0.9 | −0.5 | −1.4 | −0.7 | −1.3 | −0.8 |

## Latency (Best performance mode, not pinned)

The figures are ms per image, inference only, as the median of 3 interleaved rounds × 50 images.

| runtime | 26n | 11n | 26s | 11s | 26m | 11m |
|---|---|---|---|---|---|---|
| PyTorch CPU | 52.7 | 49.8 | 95.2 | 92.3 | 200.8 | 198.3 |
| ONNX CPU | 16.7 | 20.8 | 46.0 | 48.5 | 131.0 | 133.7 |
| OpenVINO FP32 CPU | 20.1 | 24.9 | 58.8 | 58.6 | 171.6 | 168.8 |
| **OpenVINO INT8 CPU** | **9.8** | 10.6 | **21.8** | 21.6 | **53.8** | 53.6 |
| PyTorch CUDA FP32 | 10.9 | 10.5 | 12.4 | 11.4 | 24.7 | 23.6 |
| PyTorch CUDA FP16 | 9.4 | 7.7 | 10.8 | 8.8 | 17.0 | 16.5 |
| TensorRT FP32 † | 4.7 | 5.9 | 11.4 | 10.2 | 20.2 | 16.0 |
| **TensorRT FP16** † | **3.4** | 3.5 | **5.2** | 4.7 | **6.9** | 5.8 |

† TensorRT engines were built directly with the `tensorrt` package, not the Ultralytics exporter. The times include copying each image to the GPU. TensorRT 11 has no FP16 builder flag, so the FP16 engines come from an FP16 ONNX export built as a strongly typed network.

**End to end** (pre-processing + inference + post-processing), YOLO26n reaches **70 FPS on OpenVINO INT8 (CPU)** and **79 FPS on PyTorch CUDA FP16**.

## Windows power mode: the 2–8× trap

**YOLO26n inference, ms per image:**

| runtime | Best power efficiency | + pinned to P-cores | Best performance |
|---|---|---|---|
| PyTorch CPU | 123.8 | 52.1 | 52.7 |
| ONNX CPU | 129.8 | 37.6 | **16.7** |
| OpenVINO FP32 | 48.9 | 25.2 | 20.1 |
| OpenVINO INT8 | 22.6 | 13.2 | 9.8 |
| CUDA FP16 | 21.2 | 10.9 | 9.4 |

- **Mechanism:** a freshly started process runs fast. A few seconds later, in *Best power efficiency*, Windows moves it to slower cores and clocks. It happens even if the process is just sleeping, and no ML library is involved.
- **Pinning to the P-cores recovers most of it, but not all**, because the P-cores also clock down in that mode.
- **The GPU runs slow down too**, because the CPU-side work in each call slows.
- **Windows' Energy Saver can force this mode.** On this laptop the Settings page showed *"Power mode cannot be set when energy saver is on"*.
- **Run the repro yourself (10 seconds):**
  ```bash
  python ecore_repro.py         # Best power efficiency here: 2.2–3.6x slower after 3 s
  python ecore_repro.py --pin   # 1.2–1.9x
  # Best performance mode: 1.02x
  ```

## How I almost published wrong headlines (twice)

1. **The first sequential run showed YOLO26 "50–90% faster" than YOLO11 on the CPU.** YOLO26 had happened to run while the process was on fast cores, and YOLO11 after Windows demoted it. Interleaving the runs removed the gap.
2. **The 500-image subset showed YOLO26n +1.7–2.1 mAP ahead.** The full 5,000-image set says +0.6.

**If you benchmark on a hybrid-CPU laptop:** use *Best performance*, stay plugged in, interleave models, and use the full validation set.

## Laptop vs the model card (YOLO26, ms)

| | n | s | m |
|---|---|---|---|
| card: ONNX CPU (EC2 Xeon) | 38.9 | 87.2 | 220.0 |
| laptop: ONNX CPU | 16.7 | 46.0 | 131.0 |
| card: TensorRT FP16 (T4) | 1.7 | 2.5 | 4.7 |
| laptop: TensorRT FP16 (RTX 4060 Laptop, incl. upload) | 3.4 | 5.2 | 6.9 |

## Caveats

- **One laptop.** Please run `ecore_repro.py` and the scripts on yours and post the results.
- **Latency is the median of 3 interleaved rounds** of 50 images. The worst round-to-round spread was 1.6×.
- **TensorRT times include the copy to the GPU**, so they are not comparable 1:1 with the card's numbers.
- **INT8 used Ultralytics' default OpenVINO/NNCF calibration** on 500 images.
- **Accuracy uses Ultralytics `val` defaults** (conf 0.001, end-to-end outputs).
- **The Ultralytics 8.4.78 TensorRT exporter auto-installs `nvidia-modelopt`.** On this Windows Python that install half-removed `onnx` and tried to replace torch. The scripts set `YOLO_AUTOINSTALL=False`, and TensorRT is built directly instead.

## Reproduce

```bash
pip install -r requirements.txt
# data/coco: COCO val2017 images (images.cocodataset.org) + YOLO labels (Ultralytics coco2017labels.zip)
# data/coco500: the 500 images in data_list/pick.txt (INT8 calibration + quick checks)
py -3.12 bench.py                    # exports (ONNX, OpenVINO FP32/INT8) + first-pass numbers
py -3.12 fullval.py                  # full-val accuracy + held-out INT8 accuracy
py -3.12 retime.py --unpinned        # interleaved latency (run in Best performance)
py -3.12 trt_bench.py                # TensorRT FP32/FP16, built directly
py -3.12 ecore_repro.py              # power-mode repro
```

**Raw data** is in `results/`:
- `fullval.jsonl`: full-val and held-out accuracy.
- `retime_unpinned_bestperf.jsonl`: latency tables.
- `retime_*_powersaver.jsonl`: power-saver runs.
- `tensorrt.jsonl`: TensorRT timings.
- `results.jsonl`: first pass. Its latency is invalid, see above.
