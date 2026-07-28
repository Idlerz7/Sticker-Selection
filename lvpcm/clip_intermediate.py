"""Local Hugging Face CLIP intermediate-state extraction."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from transformers import CLIPFeatureExtractor, CLIPModel

from .vpd import multi_vpd, single_vpd


class CLIPIntermediateExtractor:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.processor = CLIPFeatureExtractor.from_pretrained(model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def extract(self, images: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        result = self.model.vision_model(pixel_values=pixels, output_hidden_states=True, return_dict=True)
        hidden = result.hidden_states
        if len(hidden) != 13:
            raise ValueError("expected 13 CLIP hidden states, got %d" % len(hidden))
        return single_vpd(hidden).cpu().float(), multi_vpd(hidden).cpu().float()
