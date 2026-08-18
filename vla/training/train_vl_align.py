"""
Stage 1 Training Engine: Multi-Embodiment Fine-Grained VL Alignment (100% GPU-Resident Edition)
"""
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, List, Dict, Any

from ..configs.config import VLAConfig, get_default_config
from ..models.openvla_alignflow import OpenVLAAlignFlow
from ..data.embodied_dataset import (
    EmbodiedVLADataset,
    GPUResidentTensorDataset,
    create_synthetic_embodied_dataset,
)


def run_stage1_vl_alignment(
    model: Optional[OpenVLAAlignFlow] = None,
    config: Optional[VLAConfig] = None,
    custom_dataset: Optional[EmbodiedVLADataset] = None,
    epochs: Optional[int] = None,
) -> OpenVLAAlignFlow:
    cfg = config or get_default_config()
    num_epochs = epochs or cfg.stage1_epochs
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    use_cuda = device.type == "cuda"

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if model is None:
        model = OpenVLAAlignFlow(cfg).to(device)
    else:
        model = model.to(device)

    if custom_dataset is None:
        experts, _ = create_synthetic_embodied_dataset(num_expert_trajs=60, seed=cfg.seed)
        dataset = EmbodiedVLADataset(experts, chunk_size=cfg.chunk_size, is_train=True)
    else:
        dataset = custom_dataset

    params = [p for p in list(model.backbone.parameters()) + list(model.alignment_head.parameters()) if p.requires_grad]
    optimizer = optim.AdamW(
        params,
        lr=cfg.stage1_lr,
        weight_decay=cfg.stage1_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and cfg.use_amp)

    print(f"\n====================================================================")
    print(f"🚀 Starting Stage 1: Multi-Embodiment Fine-Grained VL Alignment (RTX 4090)")
    print(f"   Epochs: {num_epochs} | Batch Size: {cfg.stage1_batch_size} | Samples: {len(dataset)} | Device: {device} | AMP: {use_cuda and cfg.use_amp}")
    print(f"====================================================================")

    # 100% GPU VRAM-Resident Fast Tensor Engine for RTX 4090
    if use_cuda and hasattr(dataset, "trajectories"):
        gpu_dataset = GPUResidentTensorDataset(
            trajectories=dataset.trajectories,
            device=device,
            chunk_size=cfg.chunk_size,
            image_size=cfg.image_size,
            vocab_size=cfg.text_vocab_size,
            text_max_length=cfg.text_max_length,
        )
        N = len(gpu_dataset)
        B_size = cfg.stage1_batch_size
        num_batches = (N + B_size - 1) // B_size

        model.train()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            total_infonce = 0.0
            total_affordance = 0.0
            perm = torch.randperm(N, device=device)

            for b in range(num_batches):
                idx = perm[b * B_size : min((b + 1) * B_size, N)]
                obs_img = gpu_dataset.obs_imgs[idx].float() / 255.0
                goal_img = gpu_dataset.goal_imgs[idx].float() / 255.0
                token_ids = gpu_dataset.token_ids[idx]
                aff_mask = gpu_dataset.affordance_masks[idx]
                emb_id = gpu_dataset.emb_ids[idx]

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=use_cuda and cfg.use_amp):
                    outputs = model.forward_stage1(
                        obs_image=obs_img,
                        goal_image=goal_img,
                        instruction=token_ids,
                        affordance_mask_gt=aff_mask,
                        embodiment_id=emb_id,
                    )
                    loss = outputs["stage1_loss"]
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=20.0, neginf=0.0)

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # In-place gradient sanitization: guarantee zero NaN/Inf in autograd graph
                for p in params:
                    if p.grad is not None:
                        torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0, out=p.grad)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params,
                    max_norm=1.0,
                )
                
                if epoch == 1 and b == 0:
                    print(f"\n[DEBUG] Batch 0 Loss: {loss.item():.4f}")
                    print(f"[DEBUG] Grad Norm: {grad_norm.item():.4f}")
                    grad_none_cnt = sum(1 for p in params if p.grad is None)
                    grad_zero_cnt = sum(1 for p in params if p.grad is not None and torch.all(p.grad == 0))
                    print(f"[DEBUG] Params total: {len(params)}, None grads: {grad_none_cnt}, Zero grads: {grad_zero_cnt}")
                    
                    param_before = params[0].clone().detach()

                scaler.step(optimizer)
                scaler.update()

                if epoch == 1 and b == 0:
                    param_after = params[0].clone().detach()
                    diff = torch.norm(param_before - param_after).item()
                    print(f"[DEBUG] Param[0] change norm after step: {diff:.6f}\n")

                total_loss += loss.item()
                total_infonce += outputs["infonce_loss"].item()
                total_affordance += outputs["affordance_loss"].item()

            scheduler.step()
            avg_loss = total_loss / max(1, num_batches)
            avg_infonce = total_infonce / max(1, num_batches)
            avg_affordance = total_affordance / max(1, num_batches)

            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                print(
                    f"[Stage 1 - Epoch {epoch:02d}/{num_epochs:02d}] "
                    f"Total Loss: {avg_loss:.4f} | InfoNCE: {avg_infonce:.4f} | Affordance KL: {avg_affordance:.4f}"
                )

    else:
        # Standard fallback for CPU
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.stage1_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        model.train()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            total_infonce = 0.0
            total_affordance = 0.0
            num_batches = 0

            for batch in dataloader:
                raw_obs = batch["obs_image"].to(device)
                raw_goal = batch["goal_image"].to(device)
                obs_img = (raw_obs.float() / 255.0) if raw_obs.dtype == torch.uint8 else raw_obs
                goal_img = (raw_goal.float() / 255.0) if raw_goal.dtype == torch.uint8 else raw_goal

                inst = batch["instruction"]
                aff_mask = batch["affordance_mask"].to(device)
                emb_id = batch["embodiment_id"].to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model.forward_stage1(
                    obs_image=obs_img,
                    goal_image=goal_img,
                    instruction=inst,
                    affordance_mask_gt=aff_mask,
                    embodiment_id=emb_id,
                )
                loss = outputs["stage1_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_infonce += outputs["infonce_loss"].item()
                total_affordance += outputs["affordance_loss"].item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, num_batches)
            avg_infonce = total_infonce / max(1, num_batches)
            avg_affordance = total_affordance / max(1, num_batches)

            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                print(
                    f"[Stage 1 - Epoch {epoch:02d}/{num_epochs:02d}] "
                    f"Total Loss: {avg_loss:.4f} | InfoNCE: {avg_infonce:.4f} | Affordance KL: {avg_affordance:.4f}"
                )

    print("✅ Stage 1 Multi-Embodiment VL Alignment Training Completed Successfully.\n")
    return model


def main():
    parser = argparse.ArgumentParser(description="Stage 1 VL Alignment Training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.device = args.device
    cfg.stage1_epochs = args.epochs

    run_stage1_vl_alignment(config=cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
