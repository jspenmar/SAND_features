# ---Stdlib---
from warnings import warn

# ---Dependencies---
import torch
import torch.nn as nn

# ---Custom---
from losses import PixelwiseContrastiveLoss


class HierarchicalContextAggregationLoss(nn.Module):
    """
    Implementation of Hierarchical Context Aggregation

    This loss combines multiple PixelwiseContextual losses with different (alpha, beta) scales.
    Given a descriptor with n-dims and n-losses scales, each loss is given n-dims//n-losses.
    Theoretically, each of these losses could also have different margins, but in practice they are usually equal.

    Attributes can be provided as a single number (same value is used for all losses) or as a list (must contain value
    to use with each loss).

    Attributes:
        n_scales (int): Number of PixelwiseContextual losses
        margins (list or float): Target margin distance between positives and negatives
        alphas (list or int): Minimum distance from original positive KeyPoint
        betas (list or int): Maximum distance from original positive KeyPoint
        n-negs (list or int): Number of negative samples to generate

    Methods:
        forward: Compute pixel-wise contrastive loss
        forward_eval: Detailed forward pass for logging
    """
    def __init__(self, n_scales=1, margins=0.5, alphas=None, betas=None, n_negs=10):
        super().__init__()
        self.n_scales = n_scales
        self.margins = margins
        self.alphas = alphas
        self.betas = betas
        self.n_negs = n_negs
        self.features, self.labels = None, None

        self._losses = self._parse_losses()
        self._has_warned = False

    def __repr__(self):
        params = (self.n_scales, self.margins, self.alphas, self.betas, self.n_negs)
        return f'{self.__class__.__qualname__}{params}'

    def __str__(self):
        return '__'.join([f'Loss {i}: {loss}' for i, loss in enumerate(self._losses)])

    @staticmethod
    def create_parser(parser):
        parser.add_argument('--n-s', default=1, type=int, help='Number of hierarchical sampling strategies')
        parser.add_argument('--margins', default=0.5, type=str, nargs='*', help='List of margins for each Scale')
        parser.add_argument('--alphas', default=None, type=str, nargs='*', help='List of alphas for each Scale')
        parser.add_argument('--betas', default=None, type=str, nargs='*', help='List of betas for each Scale')
        parser.add_argument('--n-negs', default=10, type=str, nargs='*', help='List of n-negs for each Scale')

    def _parse_losses(self):
        configs = [self.margins, self.alphas, self.betas, self.n_negs]
        configs = [(c if isinstance(c, (list, tuple)) else [c]*self.n_scales) for c in configs]

        configs = [*zip(*configs)]  # Transpose configs
        if len(configs) != self.n_scales:
            raise ValueError(f'Invalid number of configurations. ({self.n_scales} vs. {len(configs)}) ')

        return [PixelwiseContrastiveLoss(*c) for c in configs]

    def forward(self, features, labels):
        """ Compute pixel-wise contrastive loss.
        :param features: Vertically stacked feature maps (b, n-dim, h*2, w)
        :param labels: Horizontally stacked correspondence KeyPoints (b, n-kpts, 4) -> (x1, y1, x2, y2)
        :return: Loss
        """
        if features.shape[1] % self.n_scales and not self._has_warned:
            warn(f"Feature dimensions and scales are not exactly divisible. ({features.shape[1]} and {self.n_scales})")
            self._has_warned = True

        feature_chunks = torch.chunk(features, self.n_scales, dim=1)
        return sum([loss(feat, labels) for feat, loss in zip(feature_chunks, self._losses)])

    def forward_eval(self, features, labels):
        feature_chunks = torch.chunk(features, self.n_scales, dim=1)
        loss_evals = [loss.forward_eval(feat, labels) for feat, loss in zip(feature_chunks, self._losses)]
        loss, loss_evals = list(zip(*loss_evals))
        out_loss = sum(loss)

        output = {}
        for le, loss in zip(loss_evals, self._losses):
            for cat, vals in le.items():
                if cat not in output.keys():
                    output[cat] = {}
                for k, v in vals.items():
                    output[cat][f'{k}/{loss}'] = v
        return out_loss, output
