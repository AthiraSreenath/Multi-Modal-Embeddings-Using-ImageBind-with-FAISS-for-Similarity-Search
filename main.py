from ImageBind.imagebind import data
import torch
from ImageBind.imagebind.models import imagebind_model
from ImageBind.imagebind.models.imagebind_model import ModalityType
import faiss

# These assets will be present in the Github repo
text_list=["A dog", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

# Generate embeddings
with torch.no_grad():
    embeddings = model(inputs)


# Index the embeddings using FAISS
dimensions = embeddings[ModalityType.TEXT].shape[1]
l2_index = faiss.IndexFlatL2(dimensions)
index = faiss.IndexIDMap(l2_index)

# Create a list of IDs to store all results (text, image, and audio)
all_results = text_list + image_paths + audio_paths

# Create a list to store the IDs for each modality
text_ids = list(range(1, len(text_list) + 1))
vision_ids = list(range(len(text_list) + 1, len(text_list) + 1 + len(image_paths)))
audio_ids = list(range(len(text_list) + 1 + len(image_paths), len(all_results) + 1))

# Add embeddings to the index with dynamic IDs
index.add_with_ids(embeddings[ModalityType.TEXT].cpu(), text_ids)
index.add_with_ids(embeddings[ModalityType.VISION].cpu(), vision_ids)
index.add_with_ids(embeddings[ModalityType.AUDIO].cpu(), audio_ids)

# User query - What would you like to search?
user_query = input("Enter your query: ")

# Generate embeddings of user query
user_query_input = {ModalityType.TEXT: data.load_and_transform_text([user_query], device)}

with torch.no_grad():
    query_embeddings = model(user_query_input)

user_query_embeddings = query_embeddings[ModalityType.TEXT].cpu()

# Similarity search
k = 3
D, I = index.search(user_query_embeddings, k)

# Get the result ID with most similarity
result_id = I[0][0]

# Print the corresponding result
print("Result:", all_results[result_id - 1])