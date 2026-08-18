"""
Stage 2 Training Engine: Lie Group SE(3) Flow Matching Action Head (100% GPU-Resident Edition)
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


def run_stage2_flow_pretraining(
    model: OpenVLAAlignFlow,
    config: Optional[VLAConfig] = None,
    custom_dataset: Optional[EmbodiedVLADataset] = None,
    epochs: Optional[int] = None,
) -> OpenVLAAlignFlow:
    cfg = config or get_default_config()
    num_epochs = epochs or cfg.stage2_epochs
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    use_cuda = device.type == "cuda"
    model = model.to(device)

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if custom_dataset is None:
        experts, _ = create_synthetic_embodied_dataset(num_expert_trajs=60, seed=cfg.seed)
        dataset = EmbodiedVLADataset(experts, chunk_size=cfg.chunk_size, is_train=True)
    else:
        dataset = custom_dataset

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": model.flow_action_head.parameters(), "lr": cfg.stage2_lr},
            {"params": model.backbone.parameters(), "lr": cfg.stage2_lr * 0.2},
        ],
        weight_decay=cfg.stage2_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs),
        eta_min=cfg.stage2_lr * 0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and cfg.use_amp)

    print(f"\n====================================================================")
    print(f"🚀 Starting Stage 2: Lie Group SE(3) Flow Matching Action Training (RTX 4090)")
    print(f"   Epochs: {num_epochs} | Batch Size: {cfg.stage2_batch_size} | Horizon: k={cfg.chunk_size} | Device: {device} | AMP: {use_cuda and cfg.use_amp}")
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
        B_size = cfg.stage2_batch_size
        num_batches = (N + B_size - 1) // B_size

        model.train()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            total_pos_loss = 0.0
            total_rot_loss = 0.0
            perm = torch.randperm(N, device=device)

            for b in range(num_batches):
                idx = perm[b * B_size : min((b + 1) * B_size, N)]
                obs_img = gpu_dataset.obs_imgs[idx].float() / 255.0
                token_ids = gpu_dataset.token_ids[idx]
                action_chunk = gpu_dataset.action_chunks[idx]
                emb_id = gpu_dataset.emb_ids[idx]

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=use_cuda and cfg.use_amp):
                    loss_cfm, info = model.forward_stage2(
                        obs_image=obs_img,
                        instruction=token_ids,
                        action_target=action_chunk,
                        embodiment_id=emb_id,
                    )
                    loss_cfm = torch.nan_to_num(loss_cfm, nan=0.0, posinf=50.0, neginf=0.0)

                if torch.isnan(loss_cfm) or torch.isinf(loss_cfm):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss_cfm).backward()
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
                    flow_grads = [p.grad.norm().item() for p in model.flow_action_head.parameters() if p.grad is not None]
                    print(f"\n[DEBUG Stage 2] Batch 0 Loss: {loss_cfm.item():.4f}")
                    print(f"[DEBUG Stage 2] Total Grad Norm: {grad_norm.item():.4f}")
                    print(f"[DEBUG Stage 2] Flow Head Grad Norms (first 3): {[round(x, 4) for x in flow_grads[:3]]}\n")

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss_cfm.item()
                total_pos_loss += info["loss_pos"].item()
                total_rot_loss += info["loss_rot"].item()

            scheduler.step()
            avg_loss = total_loss / max(1, num_batches)
            avg_pos = total_pos_loss / max(1, num_batches)
            avg_rot = total_rot_loss / max(1, num_batches)
            current_lr = optimizer.param_groups[0]["lr"]

            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                print(
                    f"[Stage 2 - Epoch {epoch:02d}/{num_epochs:02d}] "
                    f"LR: {current_lr:.6e} | Total CFM Loss: {avg_loss:.5f} | Pos Loss: {avg_pos:.5f} | Rot SO(3) Loss: {avg_rot:.5f}"
                )

    else:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.stage2_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        model.train()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            total_pos_loss = 0.0
            total_rot_loss = 0.0
            num_batches = 0

            for batch in dataloader:
                raw_obs = batch["obs_image"].to(device)
                obs_img = (raw_obs.float() / 255.0) if raw_obs.dtype == torch.uint8 else raw_obs
                inst = batch["instruction"]
                action_chunk = batch["action_chunk"].to(device)
                emb_id = batch["embodiment_id"].to(device)

                optimizer.zero_grad(set_to_none=True)
                loss_cfm, info = model.forward_stage2(
                    obs_image=obs_img,
                    instruction=inst,
                    action_target=action_chunk,
                    embodiment_id=emb_id,
                )
                loss_cfm.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss_cfm.item()
                total_pos_loss += info["loss_pos"].item()
                total_rot_loss += info["loss_rot"].item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, num_batches)
            avg_pos = total_pos_loss / max(1, num_batches)
            avg_rot = total_rot_loss / max(1, num_batches)

            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                print(
                    f"[Stage 2 - Epoch {epoch:02d}/{num_epochs:02d}] "
                    f"Total CFM Loss: {avg_loss:.5f} | Pos Loss: {avg_pos:.5f} | Rot SO(3) Loss: {avg_rot:.5f}"
                )

    print("✅ Stage 2 Lie Group SE(3) Flow Action Pretraining Completed Successfully.\n")
    return model


def main():
    parser = argparse.ArgumentParser(description="Stage 2 CFM Pretraining")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.device = args.device
    cfg.stage2_epochs = args.epochs

    model = OpenVLAAlignFlow(cfg)
    run_stage2_flow_pretraining(model=model, config=cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
