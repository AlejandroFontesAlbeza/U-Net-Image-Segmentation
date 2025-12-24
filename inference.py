
import torch
import torch.nn as nn
from torchvision import transforms

import time
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from uNetUtils import uNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


img_path = "dataset/train/0ce66b539f52_06.jpg"
mask_path = "dataset/train_masks/0ce66b539f52_06_mask.gif"

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    model = uNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load("resources/uNetModel.pth", map_location=device))
    model.eval()

    # Transform
    transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
    ])

    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device).float()

    # Warm-up (opcional)
    with torch.no_grad():
        for _ in range(2):
            _ = model(img)

        torch.cuda.synchronize()
        start = time.time()
        pred_mask = model(img)
        torch.cuda.synchronize()
        end = time.time()
        print(end-start)


    start = time.time()
    pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
    pred_mask = (pred_mask > 0).astype(np.uint8)

    end = time.time()
    print(end - start)

    mask_color = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    
    mask_color[pred_mask == 1] = 255 
    mask_color[pred_mask == 0] = 0

    original_img = Image.open(img_path)
    original_img = original_img.resize((512,512))
    ground_truth = Image.open(mask_path)
    ground_truth = ground_truth.resize((512,512))


    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1)
    plt.title("Input Image")
    plt.imshow(original_img)
    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.imshow(ground_truth)
    plt.subplot(1,3,3)
    plt.title("Predicted Mask")
    plt.imshow(mask_color)
    plt.savefig("inference_result.png")

if __name__ == "__main__":
    main()