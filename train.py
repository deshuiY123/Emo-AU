import argparse
import datetime
import json
import logging
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW

from common.optims import LinearWarmupCosineLRScheduler
from data.dataloading import create_dataloader_merr
from model.base_model import EmotionMultimodalQwen


def setup_logger(output_dir, timestamp_str):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"train_{timestamp_str}.log")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    logger.info(f"Logging to {log_file}")
    return logger


def plot_loss_curve(loss_history, output_dir, filename="loss_curve.png"):
    if not loss_history:
        return

    epochs = list(range(len(loss_history)))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_history, marker="o", label="Training Loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()


def append_jsonl(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_compatible_state_dict(model, state_dict):
    current_state = model.state_dict()
    compatible_state = {}
    skipped = []

    for key, value in state_dict.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    return missing, unexpected, skipped


def get_trainable_state_dict(model):
    trainable_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 【优化】使用 detach() 彻底切断计算图，防止潜在的显存泄漏
            trainable_state_dict[name] = param.detach().cpu()
    return trainable_state_dict


def build_optimizer_param_groups(
    model,
    base_lr,
    temporal_lr=None,
    min_lr=1e-6,
    warmup_lr=1e-6,
    weight_decay=0.05,
):
    if temporal_lr is None:
        temporal_lr = base_lr

    base_params = []
    temporal_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "video_temporal_encoder" in name or "face_temporal_encoder" in name:
            temporal_params.append(param)
        else:
            base_params.append(param)

    param_groups = []

    if base_params:
        param_groups.append(
            {
                "params": base_params,
                "lr": base_lr,
                "init_lr": base_lr,
                "min_lr": min_lr,
                "warmup_start_lr": warmup_lr,
                "weight_decay": weight_decay,
                "name": "base",
            }
        )

    if temporal_params:
        temporal_min_lr = min_lr * (temporal_lr / base_lr) if base_lr > 0 else min_lr
        temporal_warmup_lr = (
            warmup_lr * (temporal_lr / base_lr) if base_lr > 0 else warmup_lr
        )

        param_groups.append(
            {
                "params": temporal_params,
                "lr": temporal_lr,
                "init_lr": temporal_lr,
                "min_lr": temporal_min_lr,
                "warmup_start_lr": temporal_warmup_lr,
                "weight_decay": weight_decay,
                "name": "temporal",
            }
        )

    return param_groups


def build_stage_cfg(cfg, stage_cfg):
    cfg_stage = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_stage.dataset.merr.data_type = stage_cfg.data_type
    cfg_stage.dataset.merr.ann_path = stage_cfg.ann_path
    cfg_stage.dataset.merr.task_mode = stage_cfg.get("task_mode", "reason_only")
    cfg_stage.dataset.merr.emotion_task_ratio = stage_cfg.get("emotion_task_ratio", 0.5)
    cfg_stage.dataset.merr.reason_task_ratio = stage_cfg.get("reason_task_ratio", 0.5)
    return cfg_stage


def resolve_resume_ckpt(stage_cfg, previous_ckpt):
    ckpt = stage_cfg.get("ckpt", None)
    if ckpt == "auto_prev":
        return previous_ckpt
    return ckpt


def build_model(cfg, tokenizer, device_map=None, emotion_verbalizer_loss_weight=None):
    if emotion_verbalizer_loss_weight is None:
        emotion_verbalizer_loss_weight = cfg.run.get(
            "emotion_verbalizer_loss_weight", 0.0
        )

    return EmotionMultimodalQwen(
        tokenizer=tokenizer,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        llm_model_path=cfg.model.llm_model_path,
        audio_model_path=cfg.model.audio_model_path,
        vision_model_path=cfg.model.vision_model_path,
        device_map=device_map,
        video_token_num=cfg.model.multimodal.video_token_num,
        face_token_num=cfg.model.multimodal.face_token_num,
        au_token_num=cfg.model.au_branch.token_granularity,
        audio_token_num=cfg.model.multimodal.audio_token_num,
        au_dim=cfg.model.au_branch.au_dim,
        au_hidden_dim=cfg.model.au_branch.hidden_dim,
        au_num_layers=cfg.model.au_branch.num_layers,
        au_num_heads=cfg.model.au_branch.num_heads,
        au_dropout=cfg.model.au_branch.dropout,
        emotion_verbalizer_loss_weight=emotion_verbalizer_loss_weight,
        use_au=cfg.model.au_branch.use_au,
        au_region_grouping=cfg.model.au_branch.region_grouping,
        au_token_granularity=cfg.model.au_branch.token_granularity,
    )


def train_one_stage(cfg, stage_cfg, resume_ckpt=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    lr = stage_cfg.init_lr
    lora_r = cfg.model.lora_r
    lora_alpha = cfg.model.lora_alpha
    lora_drop = cfg.model.lora_dropout
    lr_str = f"lr{lr}"

    folder_name = (
        f"{timestamp}_{stage_cfg.name}_{lr_str}_r{lora_r}_a{lora_alpha}_drop{lora_drop}"
    )
    time_output_dir = os.path.join(stage_cfg.output_dir, folder_name)
    os.makedirs(time_output_dir, exist_ok=True)

    sample_log_dir = os.path.join(time_output_dir, "run_sample")
    os.makedirs(sample_log_dir, exist_ok=True)

    metrics_path = os.path.join(time_output_dir, "train_metrics.jsonl")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    logger = setup_logger(time_output_dir, timestamp)
    logger.info(f"===== Start Stage: {stage_cfg.name} =====")
    logger.info(f"Stage Configuration:\n{OmegaConf.to_yaml(stage_cfg)}")

    device = torch.device(
        "cuda" if (cfg.run.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    logger.info(f"Using device: {device}")

    set_seed(cfg.run.seed)

    cfg_stage = build_stage_cfg(cfg, stage_cfg)
    dataloader, tokenizer = create_dataloader_merr(cfg_stage)

    emotion_loss_weight = stage_cfg.get(
        "emotion_verbalizer_loss_weight",
        cfg.run.get("emotion_verbalizer_loss_weight", 0.0),
    )
    model = build_model(
        cfg,
        tokenizer=tokenizer,
        device_map=None,
        emotion_verbalizer_loss_weight=emotion_loss_weight,
    )

    ckpt_path = resume_ckpt
    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"====== Loading Pretrained Weights from {ckpt_path} ======")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected, skipped = load_compatible_state_dict(model, state_dict)
        logger.info(
            f"Weights loaded. Missing keys: {len(missing)}. Unexpected keys: {len(unexpected)}. Skipped keys: {len(skipped)}"
        )
    else:
        logger.info("====== No checkpoint loaded, training from scratch ======")

    model.to(device)
    model.train()

    temporal_lr = stage_cfg.get("temporal_lr", stage_cfg.init_lr)
    optimizer = AdamW(
        build_optimizer_param_groups(
            model=model,
            base_lr=stage_cfg.init_lr,
            temporal_lr=temporal_lr,
            min_lr=cfg.run.min_lr,
            warmup_lr=cfg.run.warmup_lr,
            weight_decay=cfg.run.weight_decay,
        )
    )

    iters_per_epoch = cfg.run.iters_per_epoch
    if iters_per_epoch is None:
        iters_per_epoch = len(dataloader)

    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=cfg.run.max_epoch,
        iters_per_epoch=iters_per_epoch,
        min_lr=cfg.run.min_lr,
        init_lr=stage_cfg.init_lr,
        warmup_steps=stage_cfg.warmup_steps,
        warmup_start_lr=cfg.run.warmup_lr,
    )

    # 【修复 1】: 智能适配混合精度与 Scaler (默认 Qwen 推荐的 bfloat16)
    use_amp = bool(cfg.run.amp and device.type == "cuda")
    amp_dtype_str = stage_cfg.get(
        "amp_dtype", cfg.run.get("amp_dtype", "bfloat16")
    ).lower()
    pt_dtype = torch.float16 if amp_dtype_str == "float16" else torch.bfloat16

    if use_amp and pt_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")
        logger.info("AMP enabled: Using float16 with GradScaler.")
    else:
        scaler = None
        if use_amp:
            logger.info("AMP enabled: Using bfloat16 (No GradScaler needed).")

    grad_accum_steps = stage_cfg.gradient_accumulation_steps
    logger.info(f"Gradient Accumulation Steps: {grad_accum_steps}")

    loss_history = []
    last_ckpt_path = None

    for epoch in range(cfg.run.max_epoch):
        optimizer.zero_grad()
        model.train()

        epoch_samples = []
        epoch_loss_sum = 0.0
        epoch_lm_loss_sum = 0.0
        epoch_emotion_loss_sum = 0.0
        epoch_emotion_loss_count = 0
        actual_steps_in_epoch = 0

        for step, batch in enumerate(dataloader):
            if iters_per_epoch is not None and step >= iters_per_epoch:
                break

            if "image_id" in batch and "answer" in batch:
                ids = batch["image_id"]
                ans = batch["answer"]
                epoch_samples.extend(list(zip(ids, ans)))

            scheduler.step(cur_epoch=epoch, cur_step=step)

            for key in [
                "video",
                "face",
                "au",
                "audio",
                "input_ids",
                "attention_mask",
                "labels",
            ]:
                if key in batch and isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # 【修复 2】: 使用最新的 torch.amp.autocast 并明确传入 dtype
            if use_amp:
                with torch.amp.autocast("cuda", dtype=pt_dtype):
                    outputs = model(batch)
                    raw_loss = outputs.loss
                    loss = raw_loss / grad_accum_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(batch)
                raw_loss = outputs.loss
                loss = raw_loss / grad_accum_steps
                loss.backward()

            epoch_loss_sum += raw_loss.item()
            if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
                epoch_lm_loss_sum += float(outputs.lm_loss.detach().item())
            if (
                hasattr(outputs, "emotion_verbalizer_loss")
                and outputs.emotion_verbalizer_loss is not None
            ):
                epoch_emotion_loss_sum += float(
                    outputs.emotion_verbalizer_loss.detach().item()
                )
                epoch_emotion_loss_count += 1

            actual_steps_in_epoch += 1

            # 【修复 3】: 如果使用 Scaler，在 clip_grad_norm_ 之前必须调用 unscale_()
            if (step + 1) % grad_accum_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

        # 处理 Epoch 末尾剩余不足累积步数的梯度
        if actual_steps_in_epoch > 0 and actual_steps_in_epoch % grad_accum_steps != 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            logger.info(f"[{stage_cfg.name}] Final leftover gradients applied.")

        sample_file = os.path.join(sample_log_dir, f"epoch_{epoch}_samples.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            for sid, sans in epoch_samples:
                clean_ans = str(sans).replace("\n", " ").replace("\r", "")
                f.write(f"{sid}\t{clean_ans}\n")

        # 【修复 4】: lm_loss 使用总步数做分母更准确
        epoch_train_loss = epoch_loss_sum / max(actual_steps_in_epoch, 1)
        epoch_lm_loss = epoch_lm_loss_sum / max(actual_steps_in_epoch, 1)
        epoch_emotion_loss = epoch_emotion_loss_sum / max(epoch_emotion_loss_count, 1)

        epoch_train_lr = optimizer.param_groups[0]["lr"]
        epoch_lr_by_group = {
            group.get("name", f"group_{i}"): f"{group['lr']:.6e}"
            for i, group in enumerate(optimizer.param_groups)
        }

        loss_history.append(epoch_train_loss)

        epoch_metrics = {
            "stage": stage_cfg.name,
            "epoch": epoch,
            "train_lr": f"{epoch_train_lr:.6e}",
            "train_lr_by_group": epoch_lr_by_group,
            "train_loss": f"{epoch_train_loss:.6f}",
        }
        if epoch_emotion_loss_count > 0:
            epoch_metrics["lm_loss"] = f"{epoch_lm_loss:.6f}"
            epoch_metrics["emotion_verbalizer_loss"] = f"{epoch_emotion_loss:.6f}"

        append_jsonl(metrics_path, epoch_metrics)
        logger.info(json.dumps(epoch_metrics, ensure_ascii=False))

        trainable_weights = get_trainable_state_dict(model)
        epoch_ckpt_path = os.path.join(time_output_dir, f"merr_epoch{epoch}.pth")
        torch.save(trainable_weights, epoch_ckpt_path)
        last_ckpt_path = epoch_ckpt_path
        logger.info(f"Saved epoch checkpoint to {epoch_ckpt_path}")

        plot_loss_curve(loss_history, time_output_dir, filename="loss_curve.png")

    logger.info(f"===== Stage {stage_cfg.name} Finished =====")
    logger.info(f"Last checkpoint: {last_ckpt_path}")

    return last_ckpt_path


def load_cfg(config_path=None):
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config" / "train.yaml"
    return OmegaConf.load(str(config_path))


def train(config_path=None):
    cfg = load_cfg(config_path)

    bootstrap_logger = logging.getLogger(__name__)
    if not bootstrap_logger.hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    previous_ckpt = None

    if "stages" not in cfg.run or len(cfg.run.stages) == 0:
        raise ValueError("cfg.run.stages is empty. Please define stages in train.yaml")

    for stage_cfg in cfg.run.stages:
        bootstrap_logger.info(f"Preparing stage: {stage_cfg.name}")
        resume_ckpt = resolve_resume_ckpt(stage_cfg, previous_ckpt)
        previous_ckpt = train_one_stage(cfg, stage_cfg, resume_ckpt=resume_ckpt)

    bootstrap_logger.info("All stages finished.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config" / "train.yaml"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
