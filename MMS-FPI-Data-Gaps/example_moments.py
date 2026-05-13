""" Example of how to calculate moments from skymaps """
import cdflib

from skymaps.skymap import Skymap, plotMomentsTimeSeries

fp_mms1_dist = 'data/mms1/fpi/brst/l2/des-dist/2024/01/01/mms1_fpi_brst_l2_des-dist_20240101131913_v3.4.0.cdf'

with cdflib.CDF(fp_mms1_dist) as cdf_file:
    f = cdf_file.varget(f'mms1_des_dist_brst')

# Timestep Azimuth Elevation Energy
f = f.transpose(0, 3, 2, 1)
# Timestep Energy Elevation Azimuth

steps = f.shape[0]
skymap = Skymap(steps, "MMS1_skymaps", nen=f.shape[1])
skymap.skymap = f
moments = skymap.momsTS

plotMomentsTimeSeries(moments)
