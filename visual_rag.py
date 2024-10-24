import torch
import clip
from PIL import Image
import numpy as np

# Load CLIP model with ViT backbone and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess an image
image = preprocess(Image.open("example.jpg")).unsqueeze(0).to(device)

# Define the text prompt for retrieval
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

# Compute image and text features
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# Normalize features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Compute similarity (cosine similarity)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print("Similarity:", similarity.cpu().numpy())
