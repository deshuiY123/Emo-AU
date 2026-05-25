import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from data.dataloading import create_dataloader_emer
from model.base_model import EmotionMultimodalQwen
from omegaconf import OmegaConf
from tqdm import tqdm

VALID_EMOTIONS = [
    "happy",
    "sad",
    "neutral",
    "angry",
    "worried",
    "surprise",
    "fear",
    "contempt",
    "doubt",
]


def parse_two_stage_output(text):
    text = str(text).strip()
    emotion = "unknown"
    reason = text

    import re

    match = re.search(r"emotion\s*:\s*([a-zA-Z]+)", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().lower()
        if candidate in VALID_EMOTIONS:
            emotion = candidate

    reason_match = re.search(
        r"reason\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL
    )
    if reason_match:
        reason = reason_match.group(1).strip()

    return emotion, reason


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def inference():
    logger = setup_logger()

    ckpt_path = "xxxx.pth"

    base_dir = "/xxxx"
    config_path = os.path.join(base_dir, "config/train.yaml")

    eval_base_dir = "/xxxx/eval/eval_result/"

    if "checkpoints/" in ckpt_path:
        relative_part = ckpt_path.split("checkpoints/")[-1]
        relative_dir = os.path.dirname(relative_part)
    else:
        ckpt_dir = os.path.dirname(ckpt_path)
        parts = ckpt_dir.split(os.sep)
        relative_dir = os.path.join(parts[-2], parts[-1])

    final_output_dir = os.path.join(eval_base_dir, relative_dir)
    os.makedirs(final_output_dir, exist_ok=True)

    output_csv = os.path.join(final_output_dir, "result.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(config_path)
    set_seed(cfg.run.seed)

    logger.info("Creating inference dataloader...")
    loader, tokenizer = create_dataloader_emer(cfg)

    logger.info("Initializing model...")
    model = EmotionMultimodalQwen(
        tokenizer=tokenizer,
        lora_r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        llm_model_path=cfg.model.llm_model_path,
        audio_model_path=cfg.model.audio_model_path,
        vision_model_path=cfg.model.vision_model_path,
        device_map=None,
        video_token_num=cfg.model.multimodal.video_token_num,
        face_token_num=cfg.model.multimodal.face_token_num,
        au_token_num=cfg.model.multimodal.au_token_num,
        audio_token_num=cfg.model.multimodal.audio_token_num,
        au_dim=cfg.model.au_branch.au_dim,
        au_hidden_dim=cfg.model.au_branch.hidden_dim,
        au_num_layers=cfg.model.au_branch.num_layers,
        au_num_heads=cfg.model.au_branch.num_heads,
        au_dropout=cfg.model.au_branch.dropout,
    )

    logger.info(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.to(device).eval()

    results = []

    with torch.no_grad():
        for batch in tqdm(loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            generated_texts = model.generate(batch)
            video_ids = batch["image_id"]

            for vid, txt in zip(video_ids, generated_texts):
                txt = txt.strip()
                pred_emotion, pred_reason = parse_two_stage_output(txt)
                results.append(
                    {
                        "names": vid,
                        "emotion": pred_emotion,
                        "reason": pred_reason,
                        "chi_reasons": txt,
                    }
                )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    logger.info(f"Saving results to {output_csv}...")

    df = pd.DataFrame(results)
    preferred_cols = ["names", "emotion", "reason", "chi_reasons"]
    existing_cols = [c for c in preferred_cols if c in df.columns]
    if existing_cols:
        df = df[existing_cols]
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    logger.info("Done.")


if __name__ == "__main__":
    inference()
