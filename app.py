# app.py  —  Gradio interface for Hugging Face Spaces

import numpy as np
import torch
import gradio as gr

from PIL import Image, ImageOps
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

# ── ZeroGPU Support ─────────────────────────────────
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

MODEL_NAME = "facebook/dinov2-base"

if HAS_SPACES:
    DEVICE = "cuda"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading DINOv2 model on {DEVICE} (ZeroGPU: {HAS_SPACES})...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print("DINOv2 model loaded successfully.")

# ── Dataset ──────────────────────────────────────────
NPZ_PATH = Path("catalog_embeddings.npz")

if NPZ_PATH.exists():
    data = np.load(NPZ_PATH)
    CATALOG_IDS = data["ids"]
    CATALOG_KM_CODES = data["km_codes"]
    CATALOG_CATEGORIES = data["categories"]
    CATALOG_IMAGE_PATHS = data["image_paths"]
    CATALOG_EMBEDDINGS = data["embeddings"]  # (N, 768)
    print(f"Loaded catalog: {len(CATALOG_IDS)} items")
else:
    raise FileNotFoundError(
        f"catalog_embeddings.npz not found at {NPZ_PATH.resolve()}."
    )

# ── Embedding helper ─────────────────────────────────
def _generate_embedding(pil_image: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    emb = torch.nn.functional.normalize(outputs.pooler_output, p=2, dim=1)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)

if HAS_SPACES:
    @spaces.GPU
    def embed_image(pil_image: Image.Image) -> np.ndarray:
        return _generate_embedding(pil_image)
else:
    def embed_image(pil_image: Image.Image) -> np.ndarray:
        return _generate_embedding(pil_image)

# ── Search logic ─────────────────────────────────────
def search(query_image, category, num_results):
    if query_image is None:
        return [], "Please upload an image first."

    query_emb = embed_image(query_image)

    # Cosine similarity
    similarities = CATALOG_EMBEDDINGS @ query_emb
    indices = np.arange(len(similarities))

    if category and category != "All":
        cat_lower = category.lower()
        mask = np.char.lower(CATALOG_CATEGORIES) == cat_lower
        indices = indices[mask]
        similarities = similarities[mask]

    if len(indices) == 0:
        return [], f"No items found in category '{category}'."

    num_results = int(num_results)
    top_k = np.argsort(-similarities)[:num_results]

    gallery_items = []
    for idx in top_k:
        orig_idx = indices[idx]
        img_path = Path(str(CATALOG_IMAGE_PATHS[orig_idx]))
        km_code = str(CATALOG_KM_CODES[orig_idx])
        cat = str(CATALOG_CATEGORIES[orig_idx])
        score = float(similarities[idx])

        if img_path.exists():
            gallery_items.append(
                (str(img_path), f"{km_code} ({cat}) — Match: {score:.1%}")
            )

    return gallery_items, f"Found {len(gallery_items)} matching items"

# ── Gradio UI ────────────────────────────────────────
with gr.Blocks(title="Jewelry Similarity Search") as demo:

    gr.Markdown(
        """
        # 💍 Jewelry Similarity Search
        Upload a photo of a ring or bangle, and the app will find the most
        visually similar items in the catalog using **DINOv2** embeddings.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Jewelry Image",
                height=300,
            )
            category = gr.Radio(
                choices=["All", "Rings", "Bangles"],
                value="All",
                label="Category Filter",
            )
            num_results = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Number of Results",
            )
            search_btn = gr.Button(
                "🔍 Find Similar Jewelry",
                variant="primary",
                size="lg",
            )

        with gr.Column(scale=2):
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready — upload an image to search",
            )
            gallery = gr.Gallery(
                label="Similar Items",
                columns=3,
                height=500,
                object_fit="cover",
            )

    search_btn.click(
        fn=search,
        inputs=[input_image, category, num_results],
        outputs=[gallery, status_text],
    )

    input_image.change(
        fn=search,
        inputs=[input_image, category, num_results],
        outputs=[gallery, status_text],
    )

    gr.Examples(
        examples=[
            ["output/rings/KM_409.jpg"],
            ["output/rings/KM_532.jpg"],
            ["output/bangles/KM_1012.jpg"],
        ],
        inputs=input_image,
        label="Try a catalog sample",
    )

if __name__ == "__main__":
    demo.launch()
