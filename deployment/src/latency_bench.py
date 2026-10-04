"""Latency benchmark for NoMaD exploration-mode inference on a webcam (or synthetic frames).

Run from deployment/src:
    python latency_bench.py --device mps --num-samples 8 --iters 50
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../train"))
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD  # noqa: E402
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn  # noqa: E402
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def build_model(cfg, ckpt_path, device):
    enc = NoMaD_ViNT(
        obs_encoding_size=cfg["encoding_size"],
        context_size=cfg["context_size"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"],
        mha_num_attention_layers=cfg["mha_num_attention_layers"],
        mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
    )
    enc = replace_bn_with_gn(enc)
    noise_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=cfg["encoding_size"],
        down_dims=cfg["down_dims"],
        cond_predict_scale=cfg["cond_predict_scale"],
    )
    model = NoMaD(
        vision_encoder=enc,
        noise_pred_net=noise_net,
        dist_pred_net=DenseNetwork(embedding_dim=cfg["encoding_size"]),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
    return model.to(device).eval()


def to_tensor(frames, size):
    # (1, 3*(ctx+1), H, W), same layout as deployment/src/utils.transform_images
    imgs = [NORM(f.resize(size)).unsqueeze(0) for f in frames]
    return torch.cat(imgs, dim=1)


def stats(name, xs):
    a = np.array(xs) * 1000
    print(f"{name:<12} mean {a.mean():7.1f} ms | p50 {np.percentile(a, 50):7.1f} | "
          f"p95 {np.percentile(a, 95):7.1f} | max {a.max():7.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--synthetic", action="store_true", help="use random frames, no webcam")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = yaml.safe_load(open(os.path.join(HERE, "../../train/config/nomad.yaml")))
    ckpt = os.path.join(HERE, "../model_weights/nomad.pth")
    if not os.path.exists(ckpt):
        sys.exit(f"Missing weights: {ckpt}")
    size = tuple(cfg["image_size"])
    ctx = cfg["context_size"]
    n_iters = cfg["num_diffusion_iters"]

    t0 = time.perf_counter()
    model = build_model(cfg, ckpt, device)
    print(f"device={device} model load {time.perf_counter() - t0:.1f}s")

    sched = DDPMScheduler(num_train_timesteps=n_iters, beta_schedule="squaredcos_cap_v2",
                          clip_sample=True, prediction_type="epsilon")

    cap = None
    if not args.synthetic:
        import cv2
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            sys.exit("Cannot open webcam (grant camera permission to your terminal, or use --synthetic)")

    def grab():
        if cap is None:
            return Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        ok, bgr = cap.read()
        if not ok:
            sys.exit("Camera read failed")
        return Image.fromarray(bgr[:, :, ::-1])

    queue = [grab() for _ in range(ctx + 1)]
    t_cap, t_pre, t_enc, t_diff, t_tot = [], [], [], [], []

    for i in range(args.warmup + args.iters):
        s0 = time.perf_counter()
        queue.pop(0)
        queue.append(grab())
        s1 = time.perf_counter()

        obs = to_tensor(queue, size).to(device)
        goal = torch.randn((1, 3, *size), device=device)
        mask = torch.ones(1, dtype=torch.long, device=device)  # exploration: ignore goal
        sync(device)
        s2 = time.perf_counter()

        with torch.no_grad():
            cond = model("vision_encoder", obs_img=obs, goal_img=goal, input_goal_mask=mask)
            cond = cond.repeat(args.num_samples, 1) if cond.ndim == 2 else cond.repeat(args.num_samples, 1, 1)
            sync(device)
            s3 = time.perf_counter()

            act = torch.randn((args.num_samples, cfg["len_traj_pred"], 2), device=device)
            sched.set_timesteps(n_iters)
            for k in sched.timesteps:
                noise = model("noise_pred_net", sample=act, timestep=k, global_cond=cond)
                act = sched.step(model_output=noise, timestep=k, sample=act).prev_sample
            sync(device)
            s4 = time.perf_counter()

        if i >= args.warmup:
            t_cap.append(s1 - s0); t_pre.append(s2 - s1)
            t_enc.append(s3 - s2); t_diff.append(s4 - s3); t_tot.append(s4 - s0)

    print(f"\nsamples={args.num_samples} diffusion_steps={n_iters} iters={args.iters}")
    stats("capture", t_cap)
    stats("preprocess", t_pre)
    stats("encoder", t_enc)
    stats("diffusion", t_diff)
    stats("TOTAL", t_tot)
    print(f"throughput  ~{1 / np.mean(t_tot):.1f} Hz (robot config expects 4 Hz)")


if __name__ == "__main__":
    main()
