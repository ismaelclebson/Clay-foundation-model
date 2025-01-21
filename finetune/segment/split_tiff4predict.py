import os
from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def split_geotiff(image_path, chip_size, overlap):
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        meta = src.meta.copy()
        patches = []
        for y in range(0, height, chip_size - overlap):
            for x in range(0, width, chip_size - overlap):
                window = Window(x, y, min(chip_size, width - x), min(chip_size, height - y))
                transform = src.window_transform(window)
                chip = src.read(window=window)
                patches.append((chip, x, y, window.width, window.height, transform))
    return patches, meta

def predict_on_patches(model, patches):
    model.eval()
    predictions = []
    with torch.no_grad():
        for chip, x, y, w, h, transform in tqdm(patches, desc="Predicting"):
            input_tensor = torch.from_numpy(chip).unsqueeze(0).float()
            output = model(input_tensor)
            output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=False)
            pred_patch = output.argmax(dim=1).squeeze().cpu().numpy()
            predictions.append((pred_patch, x, y, w, h, transform))
    return predictions

def reconstruct_geotiff(predictions, meta, output_path):
    height, width = meta['height'], meta['width']
    mask = np.zeros((height, width), dtype=np.uint8)
    for patch, x, y, w, h, _ in predictions:
        mask[y:y + h, x:x + w] = patch
    meta.update({"count": 1, "dtype": rasterio.uint8})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(mask, 1)

def run_geotiff_prediction(model, image_path, output_path, chip_size=256, overlap=32):
    patches, meta = split_geotiff(image_path, chip_size, overlap)
    predictions = predict_on_patches(model, patches)
    reconstruct_geotiff(predictions, meta, output_path)

# Exemplo de uso:
# model = carregar_modelo_treinado()
# run_geotiff_prediction(model, "imagem_alta_resolucao.tif", "predicao_segmentada.tif")
