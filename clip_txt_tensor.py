import torch
import clip
import numpy as np
from PIL import Image

num= 6
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
CLASS_NAMES = (
    #  "foggy", "evening", "night",  "day",
    "person", "dog", "bicycle", "sports ball", "car",  "boat", "motorcycle", "truck", "baby carriage",  "bus", 
)
txt = ["a photo of the " + x for x in CLASS_NAMES]
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(txt).to(device)

with torch.no_grad():
    # image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(f'text features shape{text_features.shape}')
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
# torch.save(text_features.cpu(), './cls_tensor/uav_tensot.pt')
num_features = text_features.cpu()
torch.save(num_features, './cls_tensor/class.pt')
# num_features = text_features.cpu()[num:, :]
# num = 10 - num
# torch.save(num_features, f'./cls_tensor/10_{10-num}_{num}.pt')
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]