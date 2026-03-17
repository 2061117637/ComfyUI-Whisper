import whisper
import os
import folder_paths
import uuid
import torchaudio
import torch
import logging

import comfy.model_management as mm
import comfy.model_patcher

WHISPER_MODEL_SUBDIR = os.path.join("stt", "whisper")

logger = logging.getLogger(__name__)

WHISPER_PATCHER_CACHE = {}

class WhisperModelWrapper(torch.nn.Module):
    """
    A torch.nn.Module wrapper for Whisper models.
    This allows ComfyUI's model management to treat Whisper models like any other
    torch module, enabling device placement and memory management.
    """
    def __init__(self, model_name, download_root):
        super().__init__()
        self.model_name = model_name
        self.download_root = download_root
        self.whisper_model = None
        self.model_loaded_weight_memory = 0

    def load_model(self, device):
        """Load the Whisper model from disk to the specified device"""
        self.whisper_model = whisper.load_model(
            self.model_name,
            download_root=self.download_root,
            device=device
        )
        # Estimate model size for memory management
        model_size = sum(p.numel() * p.element_size() for p in self.whisper_model.parameters())
        self.model_loaded_weight_memory = model_size

class WhisperPatcher(comfy.model_patcher.ModelPatcher):
    """
    Custom ModelPatcher for Whisper models that integrates with ComfyUI's
    model management system for proper loading/offloading.
    """
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def patch_model(self, device_to=None, *args, **kwargs):
        """
        This method is called by ComfyUI's model manager when it's time to load
        the model onto the target device (usually the GPU). Our responsibility here
        is to ensure the model weights are loaded from disk if they haven't been already.
        """
        target_device = self.load_device

        if self.model.whisper_model is None:
            logger.info(f"Loading Whisper model '{self.model.model_name}' to {target_device}...")
            self.model.load_model(target_device)
            self.size = self.model.model_loaded_weight_memory
        else:
            logger.info(f"Whisper model '{self.model.model_name}' already in memory.")

        return super().patch_model(device_to=target_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        """
        Offload the Whisper model to free up VRAM.
        """
        if unpatch_weights:
            logger.info(f"Offloading Whisper model '{self.model.model_name}' to {device_to}...")
            self.model.whisper_model = None
            self.model.model_loaded_weight_memory = 0
            mm.soft_empty_cache()
        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)


class ApplyWhisperNode:
    languages_by_name = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'],),
            },
            "optional": {
                "language": (
                    ["auto"] +
                    [s.capitalize() for s in sorted(list(whisper.tokenizer.LANGUAGES.values())) ],
                ),
                "prompt": ("STRING", {"default":""}),
                "max_words_per_segment": ("INT", {
                    "default": 0,
                    "min": 0,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "whisper_alignment", "whisper_alignment")
    RETURN_NAMES = ("text", "segments_alignment", "words_alignment")
    FUNCTION = "apply_whisper"
    CATEGORY = "whisper"

    def _build_segment_text_from_words(self, words):
        # Keep Whisper's token-level spacing behavior when possible.
        text = "".join(word.get("word", "") for word in words).strip()
        if text:
            return text

        # Fallback for unusual cases with empty/whitespace-only tokens.
        return " ".join(word.get("word", "").strip() for word in words if word.get("word", "").strip())

    def _split_segment_by_word_limit(self, segment, max_words_per_segment):
        segment_words = [word for word in segment.get("words", []) if word.get("word", "").strip()]

        if max_words_per_segment <= 0 or len(segment_words) <= max_words_per_segment:
            return [{
                "value": segment.get("text", "").strip(),
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0)
            }]

        split_segments = []
        for idx in range(0, len(segment_words), max_words_per_segment):
            chunk = segment_words[idx: idx + max_words_per_segment]
            if len(chunk) == 0:
                continue

            start_time = chunk[0].get("start", segment.get("start", 0.0))
            end_time = chunk[-1].get("end", segment.get("end", start_time))
            if end_time < start_time:
                end_time = start_time

            split_segments.append({
                "value": self._build_segment_text_from_words(chunk),
                "start": start_time,
                "end": end_time
            })

        return split_segments if len(split_segments) > 0 else [{
            "value": segment.get("text", "").strip(),
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0)
        }]

    def apply_whisper(self, audio, model, language, prompt, max_words_per_segment=0):

        # save audio bytes from VHS to file
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        audio_save_path = os.path.join(temp_dir, f"{uuid.uuid1()}.wav")
        torchaudio.save(audio_save_path, audio['waveform'].squeeze(
            0), audio["sample_rate"])

        cache_key = model
        if cache_key not in WHISPER_PATCHER_CACHE:
            load_device = mm.get_torch_device()
            download_root = os.path.join(folder_paths.models_dir, WHISPER_MODEL_SUBDIR)
            logger.info(f"Creating Whisper ModelPatcher for {model} on device {load_device}")
            
            model_wrapper = WhisperModelWrapper(model, download_root)
            patcher = WhisperPatcher(
                model=model_wrapper,
                load_device=load_device,
                offload_device=mm.unet_offload_device(),
                size=0  # Will be set when model loads
            )
            WHISPER_PATCHER_CACHE[cache_key] = patcher

        patcher = WHISPER_PATCHER_CACHE[cache_key]

        mm.load_model_gpu(patcher)
        whisper_model = patcher.model.whisper_model

        if whisper_model is None:
            logger.error("Whisper model failed to load. Please check logs for errors.")
            raise RuntimeError(f"Failed to load Whisper model: {model}")

        transcribe_args = {"initial_prompt": prompt}

        if language != "auto":
            if ApplyWhisperNode.languages_by_name is None:
                ApplyWhisperNode.languages_by_name = {v.lower(): k for k, v in whisper.tokenizer.LANGUAGES.items()}
            transcribe_args['language'] = ApplyWhisperNode.languages_by_name[language.lower()]
        
        result = whisper_model.transcribe(audio_save_path, word_timestamps=True, **transcribe_args)

        segments = result['segments']
        segments_alignment = []
        words_alignment = []

        for segment in segments:
            # create word alignments
            segment_words = [word for word in segment.get("words", []) if word.get("word", "").strip()]
            for word in segment_words:
                word_dict = {
                    'value': word["word"].strip(),
                    'start': word["start"],
                    'end': word['end']
                }
                words_alignment.append(word_dict)

            # create segment alignments (optionally split by max words)
            split_segments = self._split_segment_by_word_limit(segment, max_words_per_segment)
            segments_alignment.extend(split_segments)

        return (result["text"].strip(), segments_alignment, words_alignment)
