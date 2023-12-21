import os
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from multilingual_clip import pt_multilingual_clip
import transformers

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Remplacement des modèles CLIP par le modèle M-CLIP
MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(MODEL_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def tokenize(texts, context_length=77, truncate=False):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=context_length, return_tensors="pt")

def encode_text(text):
    tokens = tokenize([text])
    return model.encode_text(tokens.to(model.device))

def encode_image(image, image_transform):
    image = image_transform(image).unsqueeze(0).to(model.device)
    return model.encode_image(image)
