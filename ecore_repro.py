"""10-second repro: does Windows move a long-running Python process onto E-cores (hybrid Intel CPUs)?

    python ecore_repro.py          # unpinned
    python ecore_repro.py --pin    # pinned to P-cores (--pcores N = number of logical P-core CPUs, default 12)

Times a pure-Python loop right after start-up, then again after the process has existed for a few seconds.
No ML libraries involved. A ratio well above ~1.2x unpinned (and ~1.0x pinned) means the scheduler has
moved the process to slower cores. Please post your CPU, power mode and output in the discussion.
"""
import argparse
import ctypes
import platform
import time

OVERLAYS = {  # Windows power-mode slider GUIDs
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "00000000-0000-0000-0000-000000000000": "Balanced",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}


def power_mode():
    try:
        class GUID(ctypes.Structure):
            _fields_ = [("a", ctypes.c_ulong), ("b", ctypes.c_ushort), ("c", ctypes.c_ushort), ("d", ctypes.c_ubyte * 8)]
        g = GUID()
        ctypes.WinDLL("powrprof").PowerGetEffectiveOverlayScheme(ctypes.byref(g))
        s = "%08x-%04x-%04x-%s-%s" % (g.a, g.b, g.c, bytes(g.d[:2]).hex(), bytes(g.d[2:]).hex())
        return OVERLAYS.get(s, s)
    except Exception as e:  # not Windows, or API missing
        return "unknown (%s)" % type(e).__name__


def loop_ms(n=3_000_000, reps=5):
    r = []
    for _ in range(reps):
        t = time.perf_counter()
        x = 0
        for i in range(n):
            x += i
        r.append((time.perf_counter() - t) * 1000)
    return sorted(r)[len(r) // 2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--pcores", type=int, default=12, help="number of logical CPUs that are P-cores (i7-13650HX: 12)")
    ap.add_argument("--wait", type=float, default=3.0, help="seconds to idle between the two measurements")
    a = ap.parse_args()
    if a.pin:
        import psutil
        psutil.Process().cpu_affinity(list(range(a.pcores)))
    print("CPU:", platform.processor(), "| power mode:", power_mode(), "| pinned:", a.pin)
    first = loop_ms()
    time.sleep(a.wait)
    later = loop_ms()
    print("loop at start-up: %.0f ms | after %.0f s: %.0f ms | ratio %.2fx" % (first, a.wait, later, later / first))
