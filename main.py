# ---Standard packages---
import sys
import os.path as osp

rootpath = osp.dirname(__file__)  # Parent directory of this file
if rootpath not in sys.path:
    sys.path.insert(0, rootpath)  # Prepend to path so we can use these modules

# ---External packages---
import torch
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

# ---Custom packages---
from ops import torch2np, img2torch, fmap2img
from sand import Sand, load_sand_ckpt


def main():
    n_dims = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_file = osp.join(rootpath, 'models', '10', 'ckpt_G.pt')
    img_file = osp.join(rootpath, 'images', 'sample.png')

    # Load image & convert to torch format
    img_np = imread(img_file)
    img_torch = img2torch(img_np).to(device)

    print(f'Image size (np): {img_np.shape}')
    print(f'Image size (torch): {img_torch.shape}')

    # Create & load model (single branch)
    model = Sand(n_dims).to(device)
    model.eval()
    load_sand_ckpt(model, ckpt_file)  # Check it's working by commenting out line & visualizing

    # Run inference (wrap with no_grad -> save memory + faster)
    with torch.no_grad():
        features_torch = model(img_torch).cpu()

    # Convert to image by PCAing features
    features_np = fmap2img(features_torch).squeeze(0)

    print(f'Feature size (torch): {features_torch.shape}')
    print(f'Feature size (np): {features_np.shape}')

    # Plot original image & extracted features
    _, (ax1, ax2) = plt.subplots(2, 1)
    ax1.imshow(img_np)
    ax2.imshow(features_np)
    plt.show()


if __name__ == '__main__':
    main()
