# ---Stdlib---
from os import listdir
from os.path import join

#  ---Dependencies---
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA


def get_device(device=None):
    if isinstance(device, torch.device):
        return device
    return torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_map_location():
    return None if torch.cuda.is_available() else lambda storage, loc: storage


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def torch2np(tensor):
    """
    Convert from torch tensor to numpy convention.
    If 4D -> [b, c, h, w] to [b, h, w, c]
    If 3D -> [c, h, w] to [h, w, c]

    :param tensor: Torch tensor
    :return: Numpy array
    """
    d = tensor.dim()
    perm = [0, 2, 3, 1] if d == 4 else \
           [1, 2, 0] if d == 3 else \
           [0, 1]
    return tensor.permute(perm).detach().cpu().numpy()


def np2torch(array, dtype=None):
    """
    Convert a numpy array to torch tensor convention.
    If 4D -> [b, h, w, c] to [b, c, h, w]
    If 3D -> [h, w, c] to [c, h, w]

    :param array: Numpy array
    :param dtype: Target tensor dtype
    :return: Torch tensor
    """
    d = array.ndim
    perm = [0, 3, 1, 2] if d == 4 else \
           [2, 0, 1] if d == 3 else \
           [0, 1]

    tensor = torch.from_numpy(array).permute(perm)
    return tensor.type(dtype) if dtype else tensor.float()


def img2torch(img, batched=False):
    """
    Convert single image to torch tensor convention.
    Image is normalized and converted to 4D: [1, 3, h, w]

    :param img: Numpy image
    :param batched: Return as 4D or 3D (default)
    :return: Torch tensor
    """
    img = torch.from_numpy(img.astype(np.float32)).permute([2, 0, 1])
    if img.max() > 1:
        img = img/img.max()
    if batched:
        img.unsqueeze_(0)
    return img


def fmap2img(fmap, pca=None):
    """Convert n-dimensional torch feature map to an image via PCA or normalization."""
    if fmap.dim() < 4:
        fmap.unsqueeze_(0)
    b, c, h, w = fmap.shape

    if pca is None and c == 3:
        return torch2np(norm_quantize(fmap, per_channel=True))

    pca_fn = pca.transform if pca else PCA(n_components=3).fit_transform

    pca_feats = reshape_as_vectors(fmap).cpu().numpy()
    out = [pca_fn(f).reshape(h, w, 3) for f in pca_feats]
    out = [(x - x.min()) / (x.max() - x.min()) for x in out]
    return np.stack(out, axis=0)


def fmap2pca(fmap):
    """Convert n-dimensional torch feature map to an image via PCA."""
    pca = PCA(n_components=3)

    if fmap.dim() < 4:
        fmap.unsqueeze_(0)
    b, c, h, w = fmap.shape

    pca_feats = reshape_as_vectors(fmap).cpu().numpy()
    out = pca.fit(pca_feats.reshape(-1, c))
    return out


def freeze_model(model):
    """Fix all model parameters and prevent training."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    """Make all model parameters trainable."""
    for params in model.parameters():
        params.requires_grad = True


def norm_quantize(tensor, per_channel=False):
    """Normalize between [0, 1] and quantize to 255 image levels."""
    if per_channel:
        t_min = tensor.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0].min(0, keepdim=True)[0]
        t_max = tensor.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0].max(0, keepdim=True)[0]
        norm = (tensor - t_min) / (t_max - t_min)
    else:
        norm = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    quant = torch.round(norm*255)/255
    return quant


def upsample_like(tensor, ref_tensor):
    """Upsample tensor to match ref_tensor shape."""
    return F.interpolate(tensor, size=ref_tensor.shape[-2:], mode='bilinear', align_corners=True)


def reshape_as_vectors(tensor):
    """Reshape from (b, c, h, w) to (b, h*w, c)."""
    b, c = tensor.shape[:2]
    return tensor.reshape(b, c, -1).permute(0, 2, 1)


def reshape_as_fmap(tensor, shape):
    """Reshape from (b, h*w, c) to (b, c, h, w)."""
    b, (h, w) = tensor.shape[0], shape
    return tensor.reshape(b, h, w, -1).permute(0, 3, 1, 2)


def extract_kpt_vectors(tensor, kpts, rand_batch=False):
    """
    Pick channel vectors from 2D location in tensor.
    E.g. tensor[b, :, y1, x1]

    :param tensor: Tensor to extract from [b, c, h, w]
    :param kpts: Tensor with 'n' keypoints (x, y) as [b, n, 2]
    :param rand_batch: Randomize tensor in batch the vector is extracted from
    :return: Tensor entries as [b, n, c]
    """
    batch_size, num_kpts = kpts.shape[:-1]  # [b, n]

    # Reshape as a single batch -> [b*n, 2]
    tmp_idx = kpts.contiguous().view(-1, 2).long()

    # Flatten batch number indexes  -> [b*n] e.g. [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    b_num = torch.arange(batch_size)
    b_num = b_num.repeat((num_kpts, 1)).view(-1)
    b_num = torch.sort(b_num)[0] if not rand_batch else b_num[torch.randperm(len(b_num))]

    # Perform indexing and reshape to [b, n, c]
    return tensor[b_num, :, tmp_idx[:, 1], tmp_idx[:, 0]].reshape([batch_size, num_kpts, -1])


def extract_kpt_vectors_dense(tensor, kpts_t):
    """
    Dense version of get_entries.

    :param tensor: Tensor to extract from [b, c, h, w]
    :param kpts_t: Tensor with indexes (x, y) in channels as [b, 2, h, w]
    :return: Tensor entries as [b, c, h, w]
    """
    b, c, h, w = tensor.shape
    kpts = kpts_t.view(b, 2, -1).permute([0, 2, 1])
    entries = extract_kpt_vectors(tensor, kpts)
    return entries.permute([0, 2, 1]).view(b, c, h, w)


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
    files = [f for f in listdir(ckpt_dir) if f.endswith('.pt')]
    try:
        ckpt_file = next(f for f in sorted(files, reverse=not reverse) if f not in ignore)
    except StopIteration:
        ckpt_file = None

    return join(ckpt_dir, ckpt_file) if ckpt_file else None


def random_sample(tensor, n_sqrt):
    """ Randomly sample n**2 vectors from a tensor and arrange into a square tensor."""
    n = n_sqrt ** 2

    b, c, h, w = tensor.shape
    x, y = torch.randint(high=w, size=(b, n), dtype=torch.int), torch.randint(high=h, size=(b, n), dtype=torch.int)
    kpts = torch.stack((x, y), dim=-1)

    entries = extract_kpt_vectors(tensor, kpts)
    return entries.reshape(b, n_sqrt, n_sqrt, c).permute(0, -1, 1, 2), kpts

