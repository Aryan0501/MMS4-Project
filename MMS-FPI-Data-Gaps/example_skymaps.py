""" Example of generators for real and artificial occlusions """
from skymaps.generator import SkymapDataset, plotSkymap
from skymaps.helper import findCDFs

# Read an MMS1 Burst file and apply an artificial occlusion
print("MMS1")
fps = findCDFs('data/mms1/fpi/brst/l2/des-dist')
generator = SkymapDataset(fps,
                          shuffle=False,
                          detect_occlusion=False)
for x, m, f in generator:
    print(f"Skymaps: {f.shape}")
    print(f"Occluded Skymaps: {x.shape}")
    print(f"Mask: {m.shape}")

    plotSkymap(x, m, f, 0)

# Read an MMS4 Burst file and get the real occlusion
print("MMS4")
fps = findCDFs('data/mms4/fpi/brst/l2/des-dist')
generator = SkymapDataset(fps,
                          shuffle=False,
                          detect_occlusion=True)
for x, m, f in generator:
    print(f"Skymaps: {f.shape}")
    print(f"Occluded Skymaps: {x.shape}")
    print(f"Mask: {m.shape}")

    plotSkymap(x, m, f, 0)
