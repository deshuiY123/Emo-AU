import logging
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

from data.utils.media_io import (
    AU_COLUMNS_17,
    load_audio_fixed_length,
    load_emotion2vec_feature_fixed_length,
    load_video_face_au_synced,
)

logger = logging.getLogger(__name__)


class EMER_Dataset(Dataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        ann_path,
        vis_root="/xxxx/MERR/video",
        audio_root="/xxxx/MERR/audio",
        face_root="/xxxx/MERR/openface",
        face_au_root="/xxxx/MERR/openface_au",
        sampled_video_frames=24,
        sampled_face_au_frames=64,
        max_audio_len=16000 * 10,
        face_au_sampling_strategy="uniform",
        audio_model_path="/xxxx/model/TencentGameMate/chinese-hubert-large",
        audio_input_type="waveform",
        emotion2vec_feature_root=None,
        max_audio_feature_frames=500,
        emotion2vec_dim=1024,
    ):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.audio_input_type = str(audio_input_type or "waveform").lower()
        self.emotion2vec_feature_root = emotion2vec_feature_root
        self.max_audio_feature_frames = int(max_audio_feature_frames)
        self.emotion2vec_dim = int(emotion2vec_dim)
        if self.audio_input_type == "emotion2vec":
            self.audio_processor = None
        else:
            self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                audio_model_path
            )

        self.vis_root = vis_root
        self.audio_root = audio_root
        self.face_root = face_root
        self.face_au_root = face_au_root

        self.sampled_video_frames = sampled_video_frames
        self.sampled_face_au_frames = sampled_face_au_frames
        self.max_audio_len = max_audio_len

        self.au_columns = AU_COLUMNS_17
        self.face_au_sampling_strategy = face_au_sampling_strategy

        self.modality_prompt_prefix_pool = [
            (
                "<videofeature>...</videofeature> for global visual context and motion, "
                "<facefeature>...</facefeature> for local facial appearance, "
                "<aufeature>...</aufeature> for facial action unit dynamics, and "
                "<audiofeature>...</audiofeature> for vocal and prosodic cues. "
                "Use all modalities jointly for emotion understanding."
            )
        ]

        self.emotion_labels = [
            "happy",
            "sad",
            "angry",
            "worried",
            "surprise",
        ]
        self.emotion_label_text = ", ".join(self.emotion_labels)

        self.reason_instruction_pool = [
            "Analyze the multimodal evidence and explain the person's emotional state. Give the inferred emotion and the supporting reasons.",
        ]

        self.sample_names = []
        logger.info(f"Loading sample list from: {ann_path}")
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    self.sample_names.append(parts[0])
        logger.info(f"Total samples to inference: {len(self.sample_names)}")

        self.character_lines = pd.read_csv(
            "/xxxx/MERR/transcription_en_all.csv"
        )

    def _build_two_stage_format_instruction(self):
        return (
            "You must answer using exactly the following two-line format:\n"
            "Emotion: <one label from the candidate list>\n"
            "Reason: <a concise explanation grounded in the multimodal evidence>"
        )

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        video_name = self.sample_names[index]

        video_path_mp4 = os.path.join(self.vis_root, video_name + ".mp4")
        video_path_avi = os.path.join(self.vis_root, video_name + ".avi")
        if os.path.exists(video_path_mp4):
            video_path = video_path_mp4
        elif os.path.exists(video_path_avi):
            video_path = video_path_avi
        else:
            raise FileNotFoundError(
                f"Video file not found for {video_name} under {self.vis_root}"
            )

        # ------------------------------
        # 全局视频：单独均匀采样
        # ------------------------------
        # video_frames = load_video_uniform(
        #     video_path=video_path, sampled_video_frames=self.sampled_video_frames
        # )

        video_frames, face_frames, au_tensor = load_video_face_au_synced(
            sample_key=video_name,
            video_path=video_path,
            face_root=self.face_root,
            face_au_root=self.face_au_root,
            sampled_frames=self.sampled_face_au_frames,
            au_columns=self.au_columns,
            image_size=(224, 224),
            sampling_strategy=self.face_au_sampling_strategy,
        )

        video_tensors = [self.vis_processor(f) for f in video_frames]
        video_input = torch.stack(video_tensors, dim=0)

        # ------------------------------
        # face + AU：success==1，同步采样
        # ------------------------------
        # face_frames, au_tensor = load_face_au_synced(
        #     sample_key=video_name,
        #     face_root=self.face_root,
        #     face_au_root=self.face_au_root,
        #     sampled_face_au_frames=self.sampled_face_au_frames,
        #     au_columns=self.au_columns,
        #     sampling_strategy=self.face_au_sampling_strategy,
        # )
        face_tensors = [self.vis_processor(f) for f in face_frames]
        face_input = torch.stack(face_tensors, dim=0)

        # ------------------------------
        # 音频
        # ------------------------------
        if self.audio_input_type == "emotion2vec":
            feature_path = os.path.join(
                self.emotion2vec_feature_root, video_name + ".npy"
            )
            audio_input = load_emotion2vec_feature_fixed_length(
                feature_path=feature_path,
                max_audio_feature_frames=self.max_audio_feature_frames,
                feature_dim=self.emotion2vec_dim,
            )
        else:
            wave_path = os.path.join(self.audio_root, video_name + ".wav")
            audio_input = load_audio_fixed_length(
                audio_processor=self.audio_processor,
                wav_path=wave_path,
                max_audio_len=self.max_audio_len,
            )

        instruction_pool = self.reason_instruction_pool[0]

        try:
            sentence = self.character_lines.loc[
                self.character_lines["name"] == video_name, "sentence"
            ].values[0]
        except Exception:
            sentence = "..."

        format_instruction = self._build_two_stage_format_instruction()
        modality_prompt = self.modality_prompt_prefix_pool[0]
        instruction_pool = (
            "First determine the person's emotion label, then explain the supporting evidence. "
            f"Choose exactly one emotion label from: {self.emotion_label_text}. "
            "Use the provided video features, face features, action units features, audio features and visual context as evidence. "
            + format_instruction
        )
        character_line = "The person in video says: '{}'. ".format(sentence)
        instruction = "<FeatureHere> {} {}".format(
            character_line,
            # modality_prompt,
            instruction_pool,
        )

        return {
            "video": video_input,
            "face": face_input,
            "au": au_tensor,
            "audio": audio_input,
            "instruction_input": instruction,
            "image_id": video_name,
        }
