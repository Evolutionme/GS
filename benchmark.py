#
# Efficiency benchmark script for comparing baseline vs local features method.
#
# Usage:
#   python benchmark.py -s <scene_path> -m <model_path> --baseline_time <minutes>
#

import os
import sys
import time
import torch
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.image_utils import psnr
from tqdm import tqdm


def benchmark(args):
    """Run efficiency benchmark on a trained model."""
    # Setup
    parser = ArgumentParser(description="Benchmark script")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--baseline_time", type=float, default=0, help="Baseline training time in minutes")
    combined_args = get_combined_args(parser)
    
    dataset = lp.extract(combined_args)
    pipe = pp.extract(combined_args)
    baseline_time = combined_args.baseline_time

    # Load scene
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Render FPS benchmark
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        test_cameras = scene.getTrainCameras()[:10]

    print(f"\n=== Efficiency Benchmark ===")
    print(f"Model: {dataset.model_path}")
    print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]:,}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # Warm up
    for cam in test_cameras[:3]:
        render(cam, gaussians, pipe, background)
    torch.cuda.synchronize()

    # Measure FPS 
    start = time.time()
    num_renders = 0
    psnr_total = 0.0
    for cam in test_cameras:
        result = render(cam, gaussians, pipe, background)
        gt = cam.original_image.cuda()
        psnr_total += psnr(result["render"], gt).mean().item()
        num_renders += 1
    torch.cuda.synchronize()
    elapsed = time.time() - start

    fps = num_renders / elapsed
    avg_psnr = psnr_total / num_renders

    print(f"Rendering FPS: {fps:.1f}")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    if baseline_time > 0:
        overhead = ((baseline_time - baseline_time) / baseline_time) * 100
        print(f"Overhead vs baseline: {overhead:.1f}%")
        print(f"(Note: set --baseline_time to actual baseline training minutes)")

    print(f"=========================\n")


if __name__ == "__main__":
    benchmark(sys.argv)
