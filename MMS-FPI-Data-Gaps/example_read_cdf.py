""" Simple example of how to open CDFs using cdflib """
import cdflib
import numpy as np

fp_mms1_dist = 'data/mms1/fpi/brst/l2/des-dist/2024/01/01/mms1_fpi_brst_l2_des-dist_20240101131913_v3.4.0.cdf'
fp_mms1_moms = 'data/mms1/fpi/brst/l2/des-moms/2024/01/01/mms1_fpi_brst_l2_des-moms_20240101131913_v3.4.0.cdf'
fp_mms4_dist = 'data/mms4/fpi/brst/l2/des-dist/2024/01/01/mms4_fpi_brst_l2_des-dist_20240101131913_v3.4.0.cdf'

with cdflib.CDF(fp_mms4_dist) as cdf_file:
    # Get the global attributes
    global_attributes = cdf_file.globalattsget()
    print("Attributes:")
    for attr, value in global_attributes.items():
        print(f"{attr}: {value}")

    # Get the variable names using cdf_info().rVariables and zVariables
    cdf_info = cdf_file.cdf_info()
    variables = cdf_info.rVariables + cdf_info.zVariables
    print("\nVariables:")
    for var in variables:
        print(var)

    # Get the attributes of each variable
    print("\nVariable Attributes:")
    for var in variables:
        var_attrs = cdf_file.varattsget(var)
        print(f"\nAttributes for {var}:")
        for attr, value in var_attrs.items():
            print(f"{attr}: {value}")

    # Get the data of each variable
    print("\nVariable Data Sizes:")
    for var in variables:
        data = cdf_file.varget(var)
        if type(data) == np.ndarray:
            print(f"{var}: {data.shape}")
        else:
            print(f"{var}: {data}")
