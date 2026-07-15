"""DuoNeRF model wrapper.

The radiance-field architecture is intentionally identical to SinNeRF. The
DuoNeRF contribution is two-reference supervision in the dataset and refreshing
semantic guidance from the currently selected reference image each iteration.
"""

from .sinnerf import SinNeRF


class DuoNeRF(SinNeRF):
    def __init__(self, hparams):
        super(DuoNeRF, self).__init__(hparams)
        self.refresh_vit_reference_each_step = True
