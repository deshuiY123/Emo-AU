import logging
import os
import random

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


class DFEW_Dataset(Dataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        ann_path,
        vis_root="/xxxx/DFEW/video",
        audio_root="/xxxx/DFEW/audio",
        face_root="/xxxx/DFEW/openface",
        face_au_root="/xxxx/DFEW/openface_au",
        sampled_video_frames=24,
        sampled_face_au_frames=100,
        max_audio_len=16000 * 10,
        mode="train",
        audio_model_path="/xxxx/model/TencentGameMate/chinese-hubert-large",
        train_transcription_path="/xxxx/DFEW/DFEW_transcription_en_train.csv",
        test_transcription_path="/xxxx/DFEW/DFEW_transcription_en_test.csv",
        face_au_sampling_strategy="uniform",
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
        self.mode = mode
        self.au_columns = AU_COLUMNS_17
        self.face_au_sampling_strategy = face_au_sampling_strategy

        self.emotions = [
            "happy",
            "sad",
            "neutral",
            "angry",
            "surprise",
            "disgust",
            "fear",
        ]
        self.emotion_label_text = ", ".join(self.emotions)
        self.emotion2id = {emo: i for i, emo in enumerate(self.emotions)}

        self.modality_prompt_prefix_pool = [
            (
                "<videofeature>...</videofeature> for global visual context and motion, "
                "<facefeature>...</facefeature> for local facial appearance, "
                "<aufeature>...</aufeature> for facial action unit dynamics, and "
                "<audiofeature>...</audiofeature> for vocal and prosodic cues. "
                "Use all modalities jointly for emotion understanding."
            ),
        ]

        self.instruction_pool = [
            (
                "First determine the person's emotion label. "
                f"Choose exactly one emotion label from: {self.emotion_label_text}. "
                "Use the provided video features, face features, action units features, audio features and visual context as evidence. "
                "Only output the emotion label, without any explanation."
            )
        ]

        logger.info(f"Loading DFEW annotations from: {ann_path}")

        self.data_list = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    self.data_list.append({"video_id": parts[0], "label": parts[2]})

        logger.info(f"Total samples: {len(self.data_list)}")

        csv_path = (
            train_transcription_path if mode == "train" else test_transcription_path
        )

        if csv_path and os.path.exists(csv_path):
            self.character_lines = pd.read_csv(csv_path, dtype={"name": str})
            self.character_lines["name"] = self.character_lines["name"].str.strip()
            self.sentence_map = dict(
                zip(self.character_lines["name"], self.character_lines["sentence"])
            )
        else:
            logger.warning(f"Transcription file not found: {csv_path}")
            self.sentence_map = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        item = self.data_list[index]
        video_name = item["video_id"]
        emotion = item["label"]

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

        # video_frames = load_video_uniform(
        #     video_path=video_path,
        #     sampled_video_frames=self.sampled_video_frames,
        #     image_size=(224, 224),
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

        # face_frames, au_tensor = load_face_au_synced(
        #     sample_key=video_name,
        #     face_root=self.face_root,
        #     face_au_root=self.face_au_root,
        #     sampled_face_au_frames=self.sampled_face_au_frames,
        #     au_columns=self.au_columns,
        #     image_size=(224, 224),
        #     sampling_strategy=self.face_au_sampling_strategy,
        # )
        face_tensors = [self.vis_processor(f) for f in face_frames]
        face_input = torch.stack(face_tensors, dim=0)

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
            wav_path = os.path.join(self.audio_root, video_name + ".wav")
            audio_input = load_audio_fixed_length(
                audio_processor=self.audio_processor,
                wav_path=wav_path,
                max_audio_len=self.max_audio_len,
            )

        if self.mode == "train":
            instruction_text = random.choice(self.instruction_pool)
        else:
            instruction_text = self.instruction_pool[0]

        modality_prompt = random.choice(self.modality_prompt_prefix_pool)

        answer = emotion
        sentence = self.sentence_map.get(video_name, "...")
        if pd.isna(sentence):
            sentence = "..."

        character_line = "The person in video says: '{}'. ".format(sentence)
        instruction = "<FeatureHere> {} {}".format(
            character_line,
            instruction_text,
        )

        return {
            "video": video_input,
            "face": face_input,
            "au": au_tensor,
            "audio": audio_input,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": video_name,
            "label": emotion,
            "emotion_label": emotion,
            "emotion_candidates": self.emotions,
        }
