import torch
import sys
import os
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import l1_loss

parser = ArgumentParser()
lp = ModelParams(parser)
pp = PipelineParams(parser)
# add local features flag
parser.add_argument('--enable_local_features', action='store_true', default=False)
args = parser.parse_args(sys.argv[1:])

dataset = lp.extract(args)
pipe = pp.extract(args)

# Force enable local features for testing
gaussians = GaussianModel(dataset.sh_degree, "default", enable_local_features=True)
scene = Scene(dataset, gaussians)

# Load checkpoint
ckpt = os.path.join(dataset.model_path, "chkpnt30000.pth")
if not os.path.exists(ckpt):
    print(f"Checkpoint not found at {ckpt}")
    sys.exit(1)
    
(model_params, iteration) = torch.load(ckpt)
gaussians.restore(model_params, dataset)

viewpoint_cam = scene.getTestCameras()[0]
bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

# Render raw
render_pkg_raw = render(viewpoint_cam, gaussians, pipe, background)
img_raw = torch.clamp(render_pkg_raw["render"], 0.0, 1.0)
gt_image = torch.clamp(viewpoint_cam.original_image.cuda(), 0.0, 1.0)

psnr_raw = psnr(img_raw, gt_image).mean().item()

# Render shadow
alpha_i = gaussians.compute_adaptive_alpha(iteration=30000, viewpoint_cam=viewpoint_cam)
knn_idx = gaussians.get_knn_idx(iteration=30000)
override_scaling, override_opacity = gaussians.get_adjusted_scaling_opacity(alpha_i, knn_idx)

render_pkg_shadow = render(viewpoint_cam, gaussians, pipe, background, override_scaling=override_scaling, override_opacity=override_opacity)
img_shadow = torch.clamp(render_pkg_shadow["render"], 0.0, 1.0)
psnr_shadow = psnr(img_shadow, gt_image).mean().item()

print(f"PSNR raw (as in current train.py evaluation): {psnr_raw:.4f}")
print(f"PSNR shadow (if we used adjusted parameters for evaluation): {psnr_shadow:.4f}")
