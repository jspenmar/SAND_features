# ---Dependencies---
import torch
import torch.nn as nn

# ---Custom---
from utils.ops import extract_kpt_vectors

Default = {'margin': 0.5, 'alpha': None, 'beta': None, 'n_neg': 10}


# noinspection PyArgumentList
class PixelwiseContrastiveLoss(nn.Module):
    """
    Implementation of "pixel-wise" contrastive loss. Contrastive loss typically compares two whole images.
            L = (Y) * (1/2 * d**2) + (1 - Y) * (1/2 * max(0, margin - d)**2)

    In this instance, we instead compare pairs of features within those images.
    Positive matches are given by ground truth correspondences between images.
    Negative matches are generated on-the-fly based on provided parameters.

    Attributes:
        margin (float): Target margin distance between positives and negatives
        alpha (int): Minimum distance from original positive KeyPoint
        beta (int): Maximum distance from original positive KeyPoint
        n-neg (int): Number of negative samples to generate

    Methods:
        forward: Compute pixel-wise contrastive loss
        forward_eval: Detailed forward pass for logging
    """
    def __init__(self, margin=0.5, alpha=None, beta=None, n_neg=10):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.n_neg = n_neg

        self._dist = nn.PairwiseDistance()

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.margin}, {self.alpha}, {self.beta}, {self.n_neg})'

    def __str__(self):
        return f'Min{self.alpha or 0}_Max{self.beta or "Inf"}'

    @staticmethod
    def create_parser(parser):
        parser.add_argument('--margin', default=0.5, help='Target distance between negative feature embeddings.')
        parser.add_argument('--alpha', default=None, type=float, help='Minimum distance from positive KeyPoint')
        parser.add_argument('--beta', default=None, type=float, help='Maximum distance from positive KeyPoint')
        parser.add_argument('--n_neg', default=10, help='Number of negative samples to generate')

    def forward(self, features, labels):
        """ Compute pixel-wise contrastive loss.
        :param features: Vertically stacked feature maps (b, n-dim, h*2, w)
        :param labels: Horizontally stacked correspondence KeyPoints (b, n-kpts, 4) -> (x1, y1, x2, y2)
        :return: Loss
        """
        source, target = torch.chunk(features, 2, dim=-2)
        source_kpts, target_kpts = torch.chunk(labels, 2, dim=-1)

        loss = self._positive_loss(source, target, source_kpts, target_kpts)[0]
        loss += self._negative_loss(source, target, source_kpts, target_kpts)[0]
        return loss

    def forward_eval(self, features, labels):
        source, target = torch.chunk(features, 2, dim=-2)
        source_kpts, target_kpts = torch.chunk(labels, 2, dim=-1)

        pos_loss, pos_dist = self._positive_loss(source, target, source_kpts, target_kpts)
        neg_loss, neg_dist = self._negative_loss(source, target, source_kpts, target_kpts)

        loss = (pos_loss+neg_loss).item()
        output = {
            'scalars': {
                'loss': loss,
                'positive': pos_dist.mean().item(),
                'negative': neg_dist.mean().item(),
            },
            'histograms': {
                'hist_positive': pos_dist.detach().cpu(),
                'hist_negative': neg_dist.detach().cpu(),
            }
        }
        return loss, output

    def _calc_distance(self, source, target, source_kpts, target_kpts):
        source_descriptors = extract_kpt_vectors(source, source_kpts).permute([0, 2, 1])
        target_descriptors = extract_kpt_vectors(target, target_kpts).permute([0, 2, 1])
        return self._dist(source_descriptors, target_descriptors)

    def _positive_loss(self, source, target, source_kpts, target_kpts):
        dist = self._calc_distance(source, target, source_kpts, target_kpts)
        loss = (dist**2).mean() / 2
        return loss, dist

    def _negative_loss(self, source, target, source_kpts, target_kpts):
        dsource_kpts, dtarget_kpts = self._generate_negative_like(source, source_kpts, target_kpts)

        dist = self._calc_distance(source, target, dsource_kpts, dtarget_kpts)
        margin_dist = (self.margin - dist).clamp(min=0.0)
        loss = (margin_dist ** 2).mean() / 2
        return loss, dist

    def _generate_negative_like(self, other, source_kpts, target_kpts):
        # Source points remain the same
        source_kpts = source_kpts.repeat([1, self.n_neg, 1])

        # Target points + offset according to method
        target_kpts = target_kpts.repeat([1, self.n_neg, 1])
        target_kpts = self._permute_negatives(target_kpts, other.shape)
        return source_kpts, target_kpts

    def _permute_negatives(self, kpts, shape):
        h, w = shape[-2:]
        # (max(h, w) - low) means that even after getting the remainder points will be further away than low
        low = self.alpha if self.alpha else 0
        high = self.beta if self.beta else (max(h, w) - low)

        # Generate random shift for each KeyPoint
        shift = torch.randint_like(kpts, low=low, high=high)
        shift *= torch.sign(torch.rand_like(shift, dtype=torch.float)-0.5).short()  # Random + or - shift

        # Initial shift to satisfy max distance
        new_kpts = kpts + shift
        new_kpts %= torch.tensor((w, h), dtype=torch.short, device=new_kpts.device)

        # Shift to satisfy min distance
        diffs = new_kpts - kpts
        diff_clamp = torch.clamp(diffs, min=-high, max=high)
        new_kpts += (diff_clamp - diffs)

        return new_kpts
