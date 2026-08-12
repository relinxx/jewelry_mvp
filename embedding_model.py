import numpy as np
import torch

from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"


if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


print(f"Using device: {DEVICE}")


processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)

model.eval()


def generate_embedding_from_image(image):
    """Generate a normalized DINOv3 embedding from a PIL image."""
    image = ImageOps.exif_transpose(image).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    embedding = outputs.pooler_output

    embedding = torch.nn.functional.normalize(
        embedding,
        p=2,
        dim=1,
    )

    embedding = (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return embedding


def generate_embedding(image_path):
    with Image.open(image_path) as image:
        return generate_embedding_from_image(image)
