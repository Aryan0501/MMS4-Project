from typing import List
import cdflib
import torch
import numpy as np
from skymaps.helper import despinCors


class SkymapDataset(torch.utils.data.Dataset):
    def __init__(self, fps: List[str], shuffle=True, detect_occlusion=False):
        super(SkymapDataset).__init__()
        self.fps = fps
        self.shuffle = shuffle
        self.occlusion_mask = np.load('skymaps/explosion.npy')
        print(self.occlusion_mask.shape)
        self.fps.sort()
        self.detect_occlusion = detect_occlusion
        self.on_epoch_end()

        self.index = 0

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, index):
        # Generate one batch of data
        # Counts vs Azimuth X Elevation X Energy X Time
        fp = str(self.fps[index])

        with cdflib.CDF(fp) as cdf_file:
            sc_num = cdf_file.cdf_info().zVariables[3][3]
            delphi = cdf_file.varget(f'mms{sc_num}_des_startdelphi_count_brst')
            f = cdf_file.varget(f'mms{sc_num}_des_dist_brst')

        # occlude data
        f, x, m = self.occlude_data(f, delphi)

        # Convert to tensor
        m = torch.from_numpy(m)
        m = m.permute(3, 0, 2, 1).type(torch.FloatTensor)

        x = torch.from_numpy(x)
        x = x.permute(3, 0, 2, 1)

        f = torch.from_numpy(f)
        f = f.permute(3, 0, 2, 1)

        return x, m, f

    def interp_delphi(self, delphi):
        # strech masks to cover the summed frames in the 4.5s window
        max_delphi = 5760

        diff = delphi[1:] - delphi[:-1]  # change between each delphi
        # repeat final change to stay the same size as original
        diff = np.append(diff, diff[-1])
        diff_neg = diff <= 0
        # replace anywhere that delphi wraps around with prev change
        diff[diff_neg] = diff[np.roll(diff_neg, -1)]

        interp_delphi = (delphi + diff) % max_delphi
        return interp_delphi.astype(int)

    def occlude_data(self, f, delphi):

        if self.detect_occlusion:
            masks = f < 0
            f[masks] = 0
            x = f.copy()
            m = masks[:, :, :, 0:1].astype(float)

        else:
            delphi = self.interp_delphi(delphi)
            occlusion_masks = np.tile(
                self.occlusion_mask, (delphi.size, 1, 1, 1))
            new_masks = despinCors(
                occlusion_masks, delphi)  # .astype(float)

            if False:  # torch.cuda.is_available():
                masks_tensor = torch.from_numpy(new_masks).to('cuda')
                torch.cumsum(masks_tensor, dim=0, out=masks_tensor)
                new_masks = masks_tensor.cpu().numpy()
                del (masks_tensor)
                torch.cuda.empty_cache()

            else:
                new_masks = np.cumsum(new_masks, axis=0)

            # new_masks = new_masks[::self.N]
            new_masks = new_masks[1:] - new_masks[:-1]
            # repeat final mask to stay the same size as original
            new_masks = np.append(new_masks, new_masks[-1:], axis=0)
            x = (1 - new_masks)*f
            m = new_masks[..., -1:].astype(float)

        return f, x, m

    def on_epoch_end(self):
        # Shuffle batches
        if self.shuffle == True:
            np.random.shuffle(self.fps)


def plotSkymap(x: torch.Tensor, m: torch.Tensor, f: torch.Tensor, t: int):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(x.numpy()[:, t, :, :].sum(axis=0))
    axs[1].imshow(m.numpy()[:, t, :, :].sum(axis=0))
    axs[2].imshow(f.numpy()[:, t, :, :].sum(axis=0))

    plt.show()
