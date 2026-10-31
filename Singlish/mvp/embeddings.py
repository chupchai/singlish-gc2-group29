"""Frozen Whisper encoder and real-audio mean pooling used by E4_02."""
import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperModel


class WhisperEmbedding:
    def __init__(self, model_name, revision):
        # CPU float32 is supported on the demo host; training used GPU bfloat16.
        # Keep CPU work bounded on small VMs and discard the unused decoder.
        torch.set_num_threads(2)
        self.extractor = WhisperFeatureExtractor.from_pretrained(model_name, revision=revision)
        model = WhisperModel.from_pretrained(
            model_name, revision=revision, dtype=torch.float32, attn_implementation="sdpa")
        self.encoder = model.get_encoder().eval()
        self.encoder.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, waveform):
        features = self.extractor(
            [waveform], sampling_rate=16000, do_normalize=False,
            device="cpu", return_tensors="pt").input_features
        hidden = self.encoder(input_features=features).last_hidden_state
        # Whisper pads to 30 seconds. Pool only the actual audio's 20 ms steps.
        steps = min(max((len(waveform) + 319) // 320, 1), hidden.shape[1])
        return hidden[0, :steps].float().mean(dim=0).cpu().numpy().astype(np.float32)
