"""Render the README / Hugging Face post charts to assets/*.png from results/*.jsonl."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

# Dark theme tokens (validated categorical pair: blue/orange on #1a1a19, CVD dE 26.8)
BG, INK, INK2, MUTED, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#8a8980", "#383835"
C26, C11 = "#3987e5", "#d95926"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold", "axes.titlesize": 13,
})


def load(name):
    return [json.loads(l) for l in open(os.path.join(HERE, "results", name))]


LAT = {(r["model"], r["config"]): r["inf_ms"] for r in load("retime_unpinned_bestperf.jsonl")}
for r in load("tensorrt.jsonl"):
    LAT[(r["model"], r["config"])] = r["inf_ms"]
SAVER = {(r["model"], r["config"]): r["inf_ms"] for r in load("retime_unpinned_powersaver.jsonl")}
PIN = {(r["model"], r["config"]): r["inf_ms"] for r in load("retime_pinned_powersaver.jsonl")}
ACC = {(r["model"], r["job"]): round(r["map50_95"] * 100 + 1e-9, 1) for r in load("fullval.jsonl")}  # round half up, as in the tables
FPS = {(r["model"], r["config"]): r["fps_wall"] for r in load("retime_unpinned_bestperf.jsonl")}

RUNTIMES = [("pytorch_cpu", "PyTorch · CPU"), ("onnx_cpu", "ONNX Runtime · CPU"), ("openvino_fp32_cpu", "OpenVINO FP32 · CPU"),
            ("openvino_int8_cpu", "OpenVINO INT8 · CPU"), ("pytorch_cuda_fp32", "PyTorch FP32 · GPU"),
            ("pytorch_cuda_fp16", "PyTorch FP16 · GPU"), ("tensorrt_fp32", "TensorRT FP32 · GPU"),
            ("tensorrt_fp16", "TensorRT FP16 · GPU")]
FOOT = "i7-13650HX + RTX 4060 Laptop · Windows 11, Best performance, AC · batch 1, 640 px · github.com/Zephyr4772/yolo26-laptop-benchmark"


def style(ax, grid_axis="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def legend(fig, y=0.93):
    h = [plt.Rectangle((0, 0), 1, 1, color=C26), plt.Rectangle((0, 0), 1, 1, color=C11)]
    fig.legend(h, ["YOLO26", "YOLO11"], loc="upper right", bbox_to_anchor=(0.985, y), ncol=2, frameon=False,
               handlelength=1.0, handleheight=1.0, fontsize=11, labelcolor=INK2)


def footer(fig, text=FOOT):
    fig.text(0.012, 0.012, text, color=MUTED, fontsize=8.5)


# ------------------------------------------------------------------ 1. hero banner
def hero():
    fig = plt.figure(figsize=(12, 3.4), dpi=150)
    fig.text(0.02, 0.86, "YOLO26 vs YOLO11 on a gaming laptop", fontsize=20, fontweight="bold")
    fig.text(0.02, 0.76, "8 runtimes · full COCO val2017 · interleaved timing · i7-13650HX + RTX 4060 Laptop",
             fontsize=11.5, color=INK2)
    d26 = ACC[("yolo26s", "full5000_cuda_fp32")] - ACC[("yolo11s", "full5000_cuda_fp32")]
    tiles = [
        ("≈", "same speed", "YOLO26 vs YOLO11, every runtime\nmostly within ±5%"),
        ("+%.1f" % d26, "mAP50-95", "YOLO26 lead on full COCO val\n(+0.6 n · +0.8 s · +0.8 m)"),
        ("%.0f FPS" % FPS[("yolo26n", "openvino_int8_cpu")], "on the CPU", "YOLO26n, OpenVINO INT8\nend to end, no GPU"),
        ("%.1f×" % (SAVER[("yolo26n", "onnx_cpu")] / LAT[("yolo26n", "onnx_cpu")]), "slower", "Windows 'Best power efficiency'\nvs 'Best performance' (ONNX)"),
    ]
    w, gap, x0 = 0.225, 0.018, 0.02
    for i, (big, unit, sub) in enumerate(tiles):
        x = x0 + i * (w + gap)
        fig.patches.append(FancyBboxPatch((x, 0.08), w, 0.56, boxstyle="round,pad=0,rounding_size=0.02",
                                          transform=fig.transFigure, facecolor="#242423", edgecolor=GRID, linewidth=1))
        fig.text(x + 0.018, 0.40, big, fontsize=26, fontweight="bold", color=INK)
        fig.text(x + 0.018, 0.31, unit, fontsize=11.5, color=INK2)
        fig.text(x + 0.018, 0.12, sub, fontsize=9.5, color=MUTED, linespacing=1.4)
    fig.savefig(os.path.join(OUT, "hero.png"))
    plt.close(fig)


# ------------------------------------------------------------------ 2. latency small multiples
def latency():
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6), dpi=150, sharey=True)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.80, bottom=0.12, wspace=0.10)
    fig.text(0.012, 0.93, "Inference latency, ms per image (lower is better)", fontsize=15, fontweight="bold")
    fig.text(0.012, 0.875, "Same laptop, interleaved rounds. The runtime moves latency 15×; the model version barely moves it.",
             fontsize=10.5, color=INK2)
    legend(fig, 0.955)
    labels = [lab for _, lab in RUNTIMES][::-1]
    keys = [k for k, _ in RUNTIMES][::-1]
    h = 0.36
    for ax, size in zip(axes, "nsm"):
        style(ax)
        for i, k in enumerate(keys):
            a, b = LAT[("yolo26" + size, k)], LAT[("yolo11" + size, k)]
            ax.barh(i + h / 2 + 0.02, a, height=h, color=C26)
            ax.barh(i - h / 2 - 0.02, b, height=h, color=C11)
            ax.text(a, i + h / 2 + 0.02, "  %.1f" % a, va="center", fontsize=8.5, color=INK2)
            ax.text(b, i - h / 2 - 0.02, "  %.1f" % b, va="center", fontsize=8.5, color=MUTED)
        ax.set_title(size.upper() + " models", loc="left", color=INK, pad=8)
        ax.set_xlim(0, max(LAT[(m + size, k)] for m in ("yolo26", "yolo11") for k in keys) * 1.22)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(labels)
    for ax in axes:
        ax.axhline(3.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    footer(fig)
    fig.savefig(os.path.join(OUT, "latency.png"))
    plt.close(fig)


# ------------------------------------------------------------------ 3. accuracy dumbbell
def accuracy():
    fig, ax = plt.subplots(figsize=(12, 3.9), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.17)
    fig.text(0.012, 0.90, "Accuracy on full COCO val2017 (5,000 images)", fontsize=15, fontweight="bold")
    fig.text(0.012, 0.82, "mAP50-95, PyTorch CUDA FP32, end-to-end outputs. YOLO26 leads at every size.", fontsize=10.5, color=INK2)
    legend(fig, 0.94)
    style(ax)
    for i, s in enumerate("nsm"):
        y = 2 - i
        a, b = ACC[("yolo26" + s, "full5000_cuda_fp32")], ACC[("yolo11" + s, "full5000_cuda_fp32")]
        ax.plot([b, a], [y, y], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        ax.scatter([b], [y], s=150, color=C11, edgecolor=BG, linewidth=2, zorder=3)
        ax.scatter([a], [y], s=150, color=C26, edgecolor=BG, linewidth=2, zorder=3)
        ax.text(a + 0.25, y, "%.1f" % a, va="center", color=INK, fontsize=10.5, fontweight="bold")
        ax.text(b - 0.25, y, "%.1f" % b, va="center", ha="right", color=INK2, fontsize=10.5)
        ax.text(a + 1.45, y, "+%.1f" % (a - b), va="center", color=C26, fontsize=10.5, fontweight="bold")
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels(["n", "s", "m"], fontsize=12, fontweight="bold")
    ax.set_xlim(37, 54.5)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("mAP50-95 (%)")
    footer(fig)
    fig.savefig(os.path.join(OUT, "accuracy.png"))
    plt.close(fig)


# ------------------------------------------------------------------ 4. speed vs accuracy
def frontier():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.14, wspace=0.18)
    fig.text(0.012, 0.92, "Speed vs accuracy: the best runtime on each device", fontsize=15, fontweight="bold")
    fig.text(0.012, 0.855, "Up and to the left is better. Accuracy is full COCO val (INT8: held-out 4,500 images).",
             fontsize=10.5, color=INK2)
    legend(fig, 0.945)
    panels = [("openvino_int8_cpu", "holdout4500_openvino_int8", "CPU · OpenVINO INT8"),
              ("tensorrt_fp16", "full5000_cuda_fp32", "GPU · TensorRT FP16")]
    for ax, (lat_key, acc_key, title) in zip(axes, panels):
        style(ax, "both")
        for fam, col in (("yolo26", C26), ("yolo11", C11)):
            xs = [LAT[(fam + s, lat_key)] for s in "nsm"]
            ys = [ACC[(fam + s, acc_key)] for s in "nsm"]
            ax.plot(xs, ys, color=col, linewidth=2, zorder=2)
            ax.scatter(xs, ys, s=90, color=col, edgecolor=BG, linewidth=2, zorder=3)
            for s, x, y in zip("nsm", xs, ys):
                dy = 0.55 if fam == "yolo26" else -0.95
                ax.text(x, y + dy, s, ha="center", fontsize=10, fontweight="bold", color=INK2)
        ax.set_title(title, loc="left", color=INK, pad=8)
        ax.set_xlabel("ms per image")
        ax.set_ylabel("mAP50-95 (%)")
        ax.margins(x=0.12, y=0.15)
    footer(fig)
    fig.savefig(os.path.join(OUT, "frontier.png"))
    plt.close(fig)


# ------------------------------------------------------------------ 5. power-mode trap
def powermode():
    rts = [("pytorch_cpu", "PyTorch · CPU"), ("onnx_cpu", "ONNX Runtime · CPU"), ("openvino_fp32_cpu", "OpenVINO FP32 · CPU"),
           ("openvino_int8_cpu", "OpenVINO INT8 · CPU"), ("pytorch_cuda_fp16", "PyTorch FP16 · GPU")]
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
    fig.subplots_adjust(left=0.17, right=0.95, top=0.70, bottom=0.15)
    fig.text(0.012, 0.92, "The Windows power-mode trap: same laptop, same model (YOLO26n)", fontsize=15, fontweight="bold")
    fig.text(0.012, 0.855, "Inference ms per image. In 'Best power efficiency', Windows moves the process to slower cores seconds after start-up.",
             fontsize=10.5, color=INK2)
    COLD, MID, HOT = "#3987e5", "#86b6ef", "#e66767"
    h = [plt.Line2D([], [], marker="o", color=c, linestyle="", markersize=9) for c in (HOT, MID, COLD)]
    fig.legend(h, ["Best power efficiency", "Best power efficiency, pinned to P-cores", "Best performance"], loc="upper left",
               bbox_to_anchor=(0.005, 0.83), ncol=3, frameon=False, fontsize=10, labelcolor=INK2)
    style(ax)
    for i, (k, lab) in enumerate(rts[::-1]):
        a, p, b = SAVER[("yolo26n", k)], PIN[("yolo26n", k)], LAT[("yolo26n", k)]
        ax.plot([b, a], [i, i], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        ax.scatter([a], [i], s=120, color=HOT, edgecolor=BG, linewidth=2, zorder=3)
        ax.scatter([p], [i], s=80, color=MID, edgecolor=BG, linewidth=2, zorder=3)
        ax.scatter([b], [i], s=120, color=COLD, edgecolor=BG, linewidth=2, zorder=4)
        ax.text(a + 3, i, "%.1f×" % (a / b), va="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_yticks(range(len(rts)))
    ax.set_yticklabels([lab for _, lab in rts[::-1]])
    ax.set_xlim(0, 145)
    ax.set_xlabel("ms per image")
    footer(fig, FOOT.replace("Best performance, AC", "AC, power mode as labelled, not pinned unless labelled"))
    fig.savefig(os.path.join(OUT, "powermode.png"))
    plt.close(fig)


if __name__ == "__main__":
    hero(); latency(); accuracy(); frontier(); powermode()
    print(sorted(os.listdir(OUT)))
