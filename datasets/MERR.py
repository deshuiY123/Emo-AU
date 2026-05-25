import json
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


class MERR_Dataset(Dataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        ann_path,
        data_type,
        vis_root="/xxxx/MERR/video",
        audio_root="/xxxx/MERR/audio",
        face_root="/xxxx/MERR/openface",
        face_au_root="/xxxx/MERR/openface_au",
        sampled_video_frames=24,
        sampled_face_au_frames=64,
        max_audio_len=16000 * 10,
        audio_model_path="/xxxx/model/TencentGameMate/chinese-hubert-large",
        task_mode="reason_only",
        emotion_task_ratio=0.5,
        reason_task_ratio=0.5,
        face_au_sampling_strategy="au_topk",
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

        self.data_type = data_type
        self.vis_root = vis_root
        self.audio_root = audio_root
        self.face_root = face_root
        self.face_au_root = face_au_root

        self.sampled_video_frames = sampled_video_frames
        self.sampled_face_au_frames = sampled_face_au_frames
        self.max_audio_len = max_audio_len

        self.task_mode = task_mode
        self.emotion_task_ratio = float(emotion_task_ratio)
        self.reason_task_ratio = float(reason_task_ratio)

        self.au_columns = AU_COLUMNS_17
        self.face_au_sampling_strategy = face_au_sampling_strategy

        self.emotion_labels = [
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
        self.emotion_label_text = ", ".join(self.emotion_labels)

        self.modality_prompt_prefix_pool = [
            (
                "<videofeature>...</videofeature> for global visual context and motion, "
                "<facefeature>...</facefeature> for local facial appearance, "
                "<aufeature>...</aufeature> for facial action unit dynamics, and "
                "<audiofeature>...</audiofeature> for vocal and prosodic cues. "
                "Use all modalities jointly for emotion understanding."
            ),
            # (
            #     "<videofeature>...</videofeature> provides global video context, "
            #     "<facefeature>...</facefeature> provides facial expression details, "
            #     "<aufeature>...</aufeature> provides structured facial action patterns, and "
            #     "<audiofeature>...</audiofeature> provides speech and prosody cues. "
            #     "Please integrate them for reasoning."
            # ),
        ]

        # self.emotion_instruction_pool = [
        #     "Determine the most appropriate emotion label for the person in the video from the following candidates: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.",
        #     # "Based on the multimodal evidence, identify which emotion category best describes the speaker: happy, sad, neutral, angry, worried, surprise, fear, contempt, or doubt.",
        #     # "Please classify the speaker's emotional state into one of these labels: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.",
        #     # "Infer the dominant emotion of the person in the video and choose one label from: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.",
        #     # "Using all available multimodal cues, decide which single emotion label best matches the person's state: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.",
        #     # "Please judge the speaker's overall emotional category from the following set: happy, sad, neutral, angry, worried, surprise, fear, contempt, doubt.",
        # ]

        # self.reason_instruction_pool = [
        #     "Analyze the multimodal evidence and explain the person's emotional state. Give the inferred emotion and the supporting reasons.",
        #     "Please infer the speaker's emotional state and explain your reasoning based on visual, facial, facial-action, audio, and contextual evidence.",
        #     "What emotion is the person expressing in the video? Please provide a brief but clear explanation grounded in multimodal cues.",
        #     "Use the multimodal evidence to determine the person's emotional state, and explain which cues support your conclusion.",
        #     "Identify the speaker's emotion and explain how facial appearance, facial action units, vocal style, and contextual information contribute to this judgment.",
        #     "Please describe the emotion-related signals in the video and infer the most likely emotional state of the speaker.",
        #     "Reason about the person's emotion by jointly considering global video context, local facial details, facial action unit patterns, and vocal cues.",
        #     "What emotional state does the person most likely have? Please explain your answer with evidence from multiple modalities.",
        # ]

        logger.info(f"ann_path: {ann_path}")

        self.ann_path = ann_path
        self.file_path = os.path.dirname(ann_path)
        self.tmp = [x.strip().split(" ") for x in open(ann_path)]

        logger.info(f"video number: {len(self.tmp)}")

        coarse_file_path = (
            "/xxxx/MERR/MERR_coarse_grained_clean.json"
        )
        with open(coarse_file_path, "r") as json_file:
            self.MERR_coarse_grained_dict = json.load(json_file)

        fine_file_path = "/xxxx/MERR/MERR_fine_grained_exNeu.json"
        with open(fine_file_path, "r") as json_file:
            self.MERR_fine_grained_dict = json.load(json_file)

        self.character_lines = pd.read_csv(
            "/xxxx/MERR/transcription_en_all.csv"
        )

    def _sample_task(self):
        if self.task_mode == "reason_only":
            return "reason"
        if self.task_mode == "emotion_only":
            return "emotion"

        total = self.emotion_task_ratio + self.reason_task_ratio
        if total <= 0:
            return "reason"
        p_emotion = self.emotion_task_ratio / total
        return "emotion" if random.random() < p_emotion else "reason"

    def _normalize_emotion(self, emotion):
        emotion_text = str(emotion).strip().lower()
        return emotion_text

    def _clean_text(self, text, default="..."):
        text = str(text).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return default
        return text

    def _build_two_stage_format_instruction(self):
        return (
            "You must answer using exactly the following two-line format:\n"
            "Emotion: <one label from the candidate list>\n"
            "Reason: <a concise explanation grounded in the multimodal evidence>"
        )

    def _build_answer_and_instruction(
        self, video_name, emotion, caption, sentence, task
    ):
        sentence = self._clean_text(sentence, default="...")
        emotion_text = self._normalize_emotion(emotion)
        modality_prompt = random.choice(self.modality_prompt_prefix_pool)
        character_line = "The person in video says: '{}'. ".format(sentence)
        format_instruction = self._build_two_stage_format_instruction()

        if task == "emotion":
            task_instruction = (
                "Determine the most appropriate emotion label for the person in the video. "
                f"Choose exactly one label from: {self.emotion_label_text}. "
                "Only the Emotion line will be used for evaluation, but keep the required two-line format. "
                + format_instruction
            )
            reason_text = "Not required for this emotion-only task."
        else:
            task_instruction = (
                "First determine the person's emotion label, then explain the supporting evidence. "
                f"Choose exactly one emotion label from: {self.emotion_label_text}. "
                "Use the provided video features, face features, action units features, audio features and visual context as evidence. "
                + format_instruction
            )
            reason_text = self._clean_text(
                str(caption).strip().rstrip(" ."),
                default=(
                    "The emotion is inferred from facial expression, facial action units, "
                    "vocal cues, speech content, and visual context."
                ),
            )

        instruction = "<FeatureHere> {} {}".format(
            character_line,
            task_instruction,
            # modality_prompt,
        )
        answer = f"Emotion: {emotion_text}\nReason: {reason_text}"

        return instruction, answer

    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp[index]
        video_name = t[0]

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
        video_input = torch.stack(
            video_tensors, dim=0
        )  # [sampled_video_frames, 3, H, W]

        # ------------------------------
        # face + AU：基于 OpenFace success==1，同步采样
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
        face_input = torch.stack(
            face_tensors, dim=0
        )  # [sampled_face_au_frames, 3, H, W]

        # ------------------------------
        # 音频
        # ------------------------------
        # wav_path = os.path.join(self.audio_root, video_name + ".wav")
        # audio_input = load_audio_fixed_length(
        #     audio_processor=self.audio_processor,
        #     wav_path=wav_path,
        #     max_audio_len=self.max_audio_len,
        # )

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

        # ------------------------------
        # 文本
        # ------------------------------
        if self.data_type == "coarse":
            reason_caption = self.MERR_coarse_grained_dict[video_name]["caption"]
            task = self._sample_task()
        else:
            reason_caption = self.MERR_fine_grained_dict[video_name][
                "smp_reason_caption"
            ]
            task = self._sample_task()

        reason_caption = self.text_processor(reason_caption)
        emotion = t[2]

        try:
            sentence = self.character_lines.loc[
                self.character_lines["name"] == video_name, "sentence"
            ].values[0]
        except Exception:
            sentence = "..."

        instruction, answer = self._build_answer_and_instruction(
            video_name=video_name,
            emotion=emotion,
            caption=reason_caption,
            sentence=sentence,
            task=task,
        )

        return {
            "video": video_input,
            "face": face_input,
            "au": au_tensor,
            "audio": audio_input,
            "instruction_input": instruction,
            "answer": answer,
            "task": task,
            "emotion_label": emotion,
            "emotion_candidates": self.emotion_labels,
            "image_id": video_name,
        }
