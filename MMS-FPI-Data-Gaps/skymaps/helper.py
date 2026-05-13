import numpy as np
from pathlib import Path


def despinCors(phi, delphi):
    # Function from Alex Barrie, modified to handle time series
    delphi_adv = (delphi + 4) % 5760
    sector = (np.floor(delphi_adv/180)).astype('int')
    newphi = np.array([np.roll(f, sect, axis=0)
                      for f, sect in zip(phi, sector)])
    return newphi


def findCDFs(root_dir, pattern='mms*_fpi_brst_l2_des-dist_*.cdf'):
    gotten_cdfs = [str(fp) for fp in Path(root_dir).rglob(pattern)]
    return gotten_cdfs
