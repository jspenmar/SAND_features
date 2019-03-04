import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA


def torch2np(tensor):
    """
    Convert from torch tensor to numpy convention.
    If 4D -> [b, c, h, w] to [b, h, w, c]
    If 3D -> [c, h, w] to [h, w, c]
    """
    d = tensor.dim()
    perm = [0, 2, 3, 1] if d == 4 \
        else [1, 2, 0] if d == 3 \
        else [0, 1]
    return tensor.permute(perm).numpy()


def img2torch(img):
    """Normalize single image and convert to torch tensor [1, 3, h, w]"""
    if img.max() > 1:
        img = img/img.max()

    img = torch.from_numpy(img.astype(np.float32)).permute([2, 0, 1])
    img.unsqueeze_(0)
    return img


def fmap2img(fmap):
    """Convert n-dimensional torch feature map to an image via PCA."""
    pca = PCA(n_components=3)

    if fmap.dim() < 4:
        fmap.unsqueeze_(0)
    b, c, h, w = fmap.shape

    pca_feats = reshape_as_vectors(fmap)
    out = [pca.fit_transform(f).reshape(h, w, 3) for f in pca_feats]
    out = [(x - x.min()) / (x.max() - x.min()) for x in out]
    return np.stack(out, axis=0)


def reshape_as_vectors(tensor):
    """Reshape from (b, c, h, w) to (b, h*w, c). """
    b, c = tensor.shape[:2]
    return tensor.reshape(b, c, -1).permute(0, 2, 1)


def reshape_as_fmap(tensor, shape):
    """Reshape from (b, h*w, c) to (b, c, h, w). """
    b, (h, w) = tensor.shape[0], shape
    return tensor.reshape(b, h, w, -1).permute(0, 3, 1, 2)


def extract_kpt_vectors(tensor, kpts):
    """
    Pick channel vectors from 2D location in tensor.
    E.g. tensor[b, :, y1, x1]

    :param tensor: Tensor to extract from [b, c, h, w]
    :param kpts: Tensor with 'n' KeyPoints (x, y) as [b, n, 2]
    :return: Tensor entries as [b, n, c]
    """
    batch_size, num_kpts = kpts.shape[:-1]  # [b, n]

    # Reshape as a single batch -> [b*n, 2]
    tmp_idx = kpts.contiguous().view(-1, 2).long()

    # Flatten batch number indexes  -> [b*n] e.g. [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    b_num = torch.arange(batch_size)
    b_num = b_num.repeat([num_kpts, 1]).view(-1)
    b_num = torch.sort(b_num)[0]

    # Perform indexing and reshape to [b, n, c]
    return tensor[b_num, :, tmp_idx[:, 1], tmp_idx[:, 0]].reshape([batch_size, num_kpts, -1])


def get_latest_ckpt(ckpt_dir, ignore=None, reverse=False):
    """
    Latest or earliest checkpoint from input directory.
    Assumes files can be sorted in a meaningful way.

    :param ckpt_dir: Directory to search in
    :param ignore: List of checkpoints to ignore (eg. corrupted?)
    :param reverse: Return earliest checkpoint
    :return: Latest checkpoint path or None
    """
    ignore = ignore or []
    files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    try:
        ckpt_file = next(f for f in sorted(files, reverse=not reverse) if f not in ignore)
    except StopIteration:
        ckpt_file = None

    return osp.join(ckpt_dir, ckpt_file) if ckpt_file else None


