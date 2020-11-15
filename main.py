# ---Stdlib---
import sys
from argparse import ArgumentParser
from pathlib import Path

# ---Dependencies---
import torch
import matplotlib.pyplot as plt
from imageio import imread

# ---Custom---
ROOT = Path(__file__).parent  # Path to repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # Prepend to path so we can use these modules

from utils import ops
from models import Sand


DEFAULT_MODEL_NAME = '3/ckpt_G'
DEFAULT_MODEL_PATH = ROOT/'ckpts'
DEFAULT_IMAGE = ROOT/'images'/'sample.png'

parser = ArgumentParser('SAND feature extraction demo')
parser.add_argument('--model-name', default=DEFAULT_MODEL_NAME, help='Name of model to load')
parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Path to directory containing models')
parser.add_argument('--image-file', default=DEFAULT_IMAGE, help='Path to image to run inference on')


def main():
    args = parser.parse_args()
    device = ops.get_device()

    ckpt_file = Path(args.model_path, args.model_name).with_suffix('.pt')
    img_file = Path(args.image_file)

    # Load image & convert to torch format
    img_np = imread(img_file)
    img_torch = ops.img2torch(img_np, batched=True).to(device)

    print(f'Image size (np): {img_np.shape}')
    print(f'Image size (torch): {img_torch.shape}')

    # Create & load model (single branch)
    model = Sand.from_ckpt(ckpt_file).to(device)
    model.eval()

    # Run inference
    with torch.no_grad():
        features_torch = model(img_torch)

    # Convert features into an images we can visualize (by PCA or normalizing)
    features_np = ops.fmap2img(features_torch).squeeze(0)

    print(f'Feature size (torch): {features_torch.shape}')
    print(f'Feature size (np): {features_np.shape}')

    # Plot original image & extracted features
    ax1, ax2 = plt.subplots(2, 1)[1]
    ax1.set_xticks([]), ax1.set_yticks([])
    ax2.set_xticks([]), ax2.set_yticks([])
    ax1.imshow(img_np)
    ax2.imshow(features_np)
    plt.show()


if __name__ == '__main__':
    main()
