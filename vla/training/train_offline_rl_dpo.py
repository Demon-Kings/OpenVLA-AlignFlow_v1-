"""
Stage 3 Training Engine: SOTA Multi-Embodiment Trajectory-DPO (100% GPU-Resident Edition)
"""
import os
import copy
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


def run_stage3_offline_rl_dpo(
    model: OpenVLAAlignFlow,
    config: Optional[VLAConfig] = None,
    custom_dataset: Optional[EmbodiedVLADataset] = None,
    epochs: Optional[int] = None,
) -> OpenVLAAlignFlow:
    cfg = config or get_default_config()
    num_epochs = epochs or cfg.stage3_epochs
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    use_cuda = device.type == "cuda"
    model = model.to(device)

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Reference policy pi_ref
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Freeze pre-trained multimodal backbone and alignment head during Stage 3 policy alignment
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.alignment_head.parameters():
        p.requires_grad = False

    if custom_dataset is None:
        experts, rejected = create_synthetic_embodied_dataset(
            num_expert_trajs=60,
            num_rejected_trajs=40,
            seed=cfg.seed,
        )
        dataset = EmbodiedVLADataset(
            trajectories=experts,
            chunk_size=cfg.chunk_size,
            is_train=True,
            rejected_trajectories=rejected,
        )
    else:
        dataset = custom_dataset

    optimizer = optim.AdamW(
        model.flow_action_head.parameters(),
        lr=cfg.stage3_lr,
        weight_decay=cfg.stage3_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs),
        eta_min=cfg.stage3_lr * 0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and cfg.use_amp)

    print(f"\n====================================================================")
    print(f"🚀 Starting Stage 3: Multi-Embodiment Trajectory-DPO Alignment (RTX 4090)")
    print(f"   Epochs: {num_epochs} | Batch Size: {cfg.stage3_batch_size} | Pairs: {len(dataset)} | Device: {device} | AMP: {use_cuda and cfg.use_amp}")
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
            rejected_trajectories=getattr(dataset, "rejected_trajectories", None),
        )
        N = len(gpu_dataset)
        B_size = cfg.stage3_batch_size
        num_batches = (N + B_size - 1) // B_size

        model.train()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            total_base_dpo = 0.0
            total_bnf = 0.0
            total_energy = 0.0
            perm = torch.randperm(N, device=device)

            for b in range(num_batches):
                idx = perm[b * B_size : min((b + 1) * B_size, N)]
                obs_img = gpu_dataset.obs_imgs[idx].float() / 255.0
                token_ids = gpu_dataset.token_ids[idx]
                act_w = gpu_dataset.action_chunks[idx]
                act_l = gpu_dataset.rejected_chunks[idx]
                emb_id = gpu_dataset.emb_ids[idx]

                B = act_w.shape[0]

                x_0 = torch.randn_like(act_w)
                t = torch.rand(B, device=device)
                t_bc = t.view(B, 1, 1)

                x_t_w = ((1.0 - t_bc) * x_0 + t_bc * act_w).detach().requires_grad_(True)
                u_t_w = act_w - x_0

                x_t_l = ((1.0 - t_bc) * x_0 + t_bc * act_l).detach().requires_grad_(True)
                u_t_l = act_l - x_0

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=use_cuda and cfg.use_amp):
                    feat = model.encode(obs_img, token_ids, emb_id)
                    context = feat["context_c"]
                    emb_feat = feat["embodiment_feat"]

                    v_w = model.flow_action_head.forward_velocity(x_t_w, t, context, emb_feat)
                    v_l = model.flow_action_head.forward_velocity(x_t_l, t, context, emb_feat)

                    with torch.no_grad():
                        ref_feat = ref_model.encode(obs_img, token_ids, emb_id)
                        ref_context = ref_feat["context_c"]
                        ref_emb_feat = ref_feat["embodiment_feat"]
                        ref_v_w = ref_model.flow_action_head.forward_velocity(x_t_w, t, ref_context, ref_emb_feat)
                        ref_v_l = ref_model.flow_action_head.forward_velocity(x_t_l, t, ref_context, ref_emb_feat)

                    dpo_metrics = model.dpo_engine(
                        v_pred_w=v_w,
                        u_target_w=u_t_w,
                        v_pred_l=v_l,
                        u_target_l=u_t_l,
                        ref_v_pred_w=ref_v_w,
                        ref_v_pred_l=ref_v_l,
                        action_w=act_w,
                        action_l=act_l,
                        x_t_w=x_t_w,
                        x_t_l=x_t_l,
                    )
                    loss = dpo_metrics["total_dpo_loss"]
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=50.0, neginf=0.0)

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # In-place gradient sanitization: guarantee zero NaN/Inf in autograd graph
                for p in model.flow_action_head.parameters():
                    if p.grad is not None:
                        torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0, out=p.grad)

                torch.nn.utils.clip_grad_norm_(
                    list(model.flow_action_head.parameters()),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                total_base_dpo += dpo_metrics["loss_base_dpo"].item()
                total_bnf += dpo_metrics["loss_bnf"].item()
                total_energy += dpo_metrics["loss_energy"].item()

            scheduler.step()
            avg_loss = total_loss / max(1, num_batches)
            avg_base_dpo = total_base_dpo / max(1, num_batches)
            avg_bnf = total_bnf / max(1, num_batches)
            avg_energy = total_energy / max(1, num_batches)

            curr_beta = model.dpo_engine.beta.item() if isinstance(model.dpo_engine.beta, torch.Tensor) else float(model.dpo_engine.beta)
            curr_lambda = model.dpo_engine.lambda_len.item() if isinstance(model.dpo_engine.lambda_len, torch.Tensor) else float(model.dpo_engine.lambda_len)

            current_lr = optimizer.param_groups[0]["lr"]
            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                print(
                    f"[Stage 3 - Epoch {epoch:02d}/{num_epochs:02d}] "
                    f"LR: {current_lr:.6e} | Total Loss: {avg_loss:.4f} | Base DPO: {avg_base_dpo:.4f} | BNF: {avg_bnf:.4f} | "
                    f"Energy Damping: {avg_energy:.4f} | Beta: {curr_beta:.3f} | Lambda_Len: {curr_lambda:.4f}"
                )

    else:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.stage3_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        model.train()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            total_base_dpo = 0.0
            total_bnf = 0.0
            total_energy = 0.0
            num_batches = 0

            for batch in dataloader:
                raw_obs = batch["obs_image"].to(device)
                obs_img = (raw_obs.float() / 255.0) if raw_obs.dtype == torch.uint8 else raw_obs
                inst = batch["instruction"]
                act_w = batch["action_chunk"].to(device)
                act_l = batch["rejected_action_chunk"].to(device)
                emb_id = batch["embodiment_id"].to(device)

                B = act_w.shape[0]

                x_0 = torch.randn_like(act_w)
                t = torch.rand(B, device=device)
                t_bc = t.view(B, 1, 1)

                x_t_w = (1.0 - t_bc) * x_0 + t_bc * act_w
                u_t_w = act_w - x_0

                x_t_l = (1.0 - t_bc) * x_0 + t_bc * act_l
                u_t_l = act_l - x_0

                optimizer.zero_grad(set_to_none=True)
                feat = model.encode(obs_img, inst, emb_id)
                context = feat["context_c"]
                emb_feat = feat["embodiment_feat"]

                v_w = model.flow_action_head.forward_velocity(x_t_w, t, context, emb_feat)
                v_l = model.flow_action_head.forward_velocity(x_t_l, t, context, emb_feat)

                with torch.no_grad():
                    ref_feat = ref_model.encode(obs_img, inst, emb_id)
                    ref_context = ref_feat["context_c"]
                    ref_emb_feat = ref_feat["embodiment_feat"]
                    ref_v_w = ref_model.flow_action_head.forward_velocity(x_t_w, t, ref_context, ref_emb_feat)
                    ref_v_l = ref_model.flow_action_head.forward_velocity(x_t_l, t, ref_context, ref_emb_feat)

                dpo_metrics = model.dpo_engine(
                    v_pred_w=v_w,
                    u_target_w=u_t_w,
                    v_pred_l=v_l,
                    u_target_l=u_t_l,
                    ref_v_pred_w=ref_v_w,
                    ref_v_pred_l=ref_v_l,
                    action_w=act_w,
                    action_l=act_l,
                )
                loss = dpo_metrics["total_dpo_loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.flow_action_head.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_base_dpo += dpo_metrics["loss_base_dpo"].item()
                total_bnf += dpo_metrics["loss_bnf"].item()
                total_energy += dpo_metrics["loss_energy"].item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(1, num_batches)
            avg_base_dpo = total_base_dpo / max(1, num_batches)
            avg_bnf = total_bnf / max(1, num_batches)
            avg_energy = total_energy / max(1, num_batches)

            curr_beta = model.dpo_engine.beta.item() if isinstance(model.dpo_engine.beta, torch.Tensor) else float(model.dpo_engine.beta)
            curr_lambda = model.dpo_engine.lambda_len.item() if isinstance(model.dpo_engine.lambda_len, torch.Tensor) else float(model.dpo_engine.lambda_len)

            if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
                print(
                    f"[Stage 3 - Epoch {epoch:02d}/{num_epochs:02d}] "
                    f"Total Loss: {avg_loss:.4f} | Base DPO: {avg_base_dpo:.4f} | BNF: {avg_bnf:.4f} | "
                    f"Energy Damping: {avg_energy:.4f} | Beta: {curr_beta:.3f} | Lambda_Len: {curr_lambda:.4f}"
                )

    print("✅ Stage 3 Multi-Embodiment Trajectory-DPO Alignment Completed Successfully.\n")
    return model


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Multi-Embodiment Trajectory-DPO Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.device = args.device
    cfg.stage3_epochs = args.epochs

    model = OpenVLAAlignFlow(cfg)
    run_stage3_offline_rl_dpo(model=model, config=cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
