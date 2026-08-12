# test_embedding.py

from embedding_model import generate_embedding

embedding = generate_embedding(
    "output/rings/KM_409.jpg"
)

print("Embedding shape:", embedding.shape)
print("First 10 values:")
print(embedding[:10])