#!/usr/bin/env python3
"""
Benchmark Whisper model sizes on this machine and print a Markdown table.

Usage:
  python scripts/benchmark.py path/to/audio.wav
  python scripts/benchmark.py audio.wav --models tiny,base --runs 5 --device cpu

Each model gets one untimed warm-up run, then --runs timed runs. Reported
latency is the median. RTF (real-time factor) is latency / audio duration;
lower is better, and below 1.0 means faster than real time.
"""

import argparse
import platform
import statistics
import subprocess
import time

import ctranslate2
from faster_whisper import WhisperModel


def hardware(device: str) -> str:
    if device == "cuda":
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return out.stdout.strip().splitlines()[0]
        except Exception:
            return "CUDA GPU"
    return f"{platform.processor() or platform.machine()} CPU"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("audio")
    parser.add_argument("--models", default="tiny,base,small,medium")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Hardware: {hardware(device)} | device={device} compute_type={compute_type} | runs={args.runs}\n")
    print("| Model | Audio (s) | Median latency (s) | RTF | Transcript |")
    print("|-------|-----------|--------------------|-----|------------|")
    for size in [m.strip() for m in args.models.split(",") if m.strip()]:
        model = WhisperModel(size, device=device, compute_type=compute_type)
        beam = 5 if size not in ("tiny", "base") else 3

        def run():
            segments, info = model.transcribe(
                args.audio, language=args.language, beam_size=beam, temperature=0.0
            )
            return " ".join(s.text.strip() for s in segments), info

        run()  # warm-up
        timings = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            text, info = run()
            timings.append(time.perf_counter() - t0)
        latency = statistics.median(timings)
        snippet = (text[:40] + "...") if len(text) > 40 else text
        print(f"| {size} | {info.duration:.1f} | {latency:.2f} | {latency / info.duration:.2f} | {snippet} |")


if __name__ == "__main__":
    main()
