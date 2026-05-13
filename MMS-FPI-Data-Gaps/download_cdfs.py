import os
import requests

urls = [
    "https://lasp.colorado.edu/mms/sdc/public/about/browse/mms1/fpi/brst/l2/des-moms/2024/01/01/mms1_fpi_brst_l2_des-moms_20240101131913_v3.4.0.cdf",
    "https://lasp.colorado.edu/mms/sdc/public/about/browse/mms1/fpi/brst/l2/des-dist/2024/01/01/mms1_fpi_brst_l2_des-dist_20240101131913_v3.4.0.cdf",
    "https://lasp.colorado.edu/mms/sdc/public/about/browse/mms4/fpi/brst/l2/des-dist/2024/01/01/mms4_fpi_brst_l2_des-dist_20240101131913_v3.4.0.cdf",
]
file_paths = [
    "data/mms1/fpi/brst/l2/des-moms/2024/01/01/mms1_fpi_brst_l2_des-moms_20240101131913_v3.4.0.cdf",
    "data/mms1/fpi/brst/l2/des-dist/2024/01/01/mms1_fpi_brst_l2_des-dist_20240101131913_v3.4.0.cdf",
    "data/mms4/fpi/brst/l2/des-dist/2024/01/01/mms4_fpi_brst_l2_des-dist_20240101131913_v3.4.0.cdf",
]


def download_file(url, file_path):
    print(f"Downloading: {url}")
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Check if the download was successful

    # Write the file to disk
    with open(file_path, 'wb') as file:
        file.write(response.content)


# Download all files
for url, file_path in zip(urls, file_paths):
    download_file(url, file_path)
