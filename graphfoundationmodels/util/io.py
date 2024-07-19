""" Auto-Download Module

This function downloads Sunsmart PV Power Plant data from the OSF database. 

"""
import pandas as pd
import requests
import os
import io

def OSF_download(
    file,
    type, 
    save_path
):
    """
    Helper function that downloads CSV files from OSF Database. The urls are already in the function so whichever one can be downloaded

    Args:
        file (string): a string that represents which file to download
        type (string): format of file to be downloaded as ('csv' or 'parquet')
        save_path (string): directory for which the file should be saved in, optional
    """
    
    urls_dict = {
        'ss2001_CBOX_1': 'https://osf.io/xcuvg/download',
        'ss2001_CBOX_2': 'https://osf.io/nm9ax/download',
        'ss2001_CBOX_3': 'https://osf.io/zw6as/download',
        'ss2002_CBOX_1': 'https://osf.io/vu4ym/download',
        'ss2002_CBOX_2': 'https://osf.io/hpgs9/download',
        'ss2002_CBOX_3': 'https://osf.io/b7a6u/download',
        'ss2003_CBOX_1': 'https://osf.io/emy7z/download',
        'ss2003_CBOX_2': 'https://osf.io/7arj5/download',
        'ss2003_CBOX_3': 'https://osf.io/36pdh/download',
        'ss2004_CBOX_1': 'https://osf.io/cug32/download',
        'ss2004_CBOX_2': 'https://osf.io/pzv9m/download',
        'ss2004_CBOX_3': 'https://osf.io/gr7cj/download',
        'ss2005_CBOX_1': 'https://osf.io/kfm2j/download',
        'ss2005_CBOX_2': 'https://osf.io/afdpx/download',
        'ss2005_CBOX_3': 'https://osf.io/afdpx/download',
        'ss2006_CBOX_1': 'https://osf.io/p3qug/download',
        'ss2006_CBOX_2': 'https://osf.io/tnj8y/download',
        'ss2006_CBOX_3': 'https://osf.io/6wk4x/download',
        'ss2007_CBOX_1': 'https://osf.io/g49t5/download',
        'ss2007_CBOX_2': 'https://osf.io/tv5xn/download',
        'ss2007_CBOX_3': 'https://osf.io/p2gvq/download',
        'ss2008_CBOX_1': 'https://osf.io/z29s8/download',
        'ss2008_CBOX_2': 'https://osf.io/7xpsv/download',
        'ss2008_CBOX_3': 'https://osf.io/wg48a/download',
        'ss2009_CBOX_1': 'https://osf.io/cp4zh/download',
        'ss2009_CBOX_2': 'https://osf.io/qe52w/download',
        'ss2009_CBOX_3': 'https://osf.io/yh3tp/download',
        'ss2010_CBOX_1': 'https://osf.io/fb73d/download',
        'ss2010_CBOX_2': 'https://osf.io/k3q2u/download',
        'ss2010_CBOX_3': 'https://osf.io/meqs4/download',
        'ss2011_CBOX_1': 'https://osf.io/x9r2q/download',
        'ss2011_CBOX_2': 'https://osf.io/hafqp/download',
        'ss2011_CBOX_3': 'https://osf.io/c2xj5/download',
        'ss2012_CBOX_1': 'https://osf.io/tk34n/download',
        'ss2012_CBOX_2': 'https://osf.io/v3xrc/download',
        'ss2012_CBOX_3': 'https://osf.io/xsv3d/download',
        'ss2013_CBOX_1': 'https://osf.io/937ty/download',
        'ss2013_CBOX_2': 'https://osf.io/j9pes/download',
        'ss2013_CBOX_3': 'https://osf.io/26uts/download',
        'ss2014_CBOX_1': 'https://osf.io/w6pn8/download',
        'ss2014_CBOX_2': 'https://osf.io/4jxt2/download',
        'ss2014_CBOX_3': 'https://osf.io/fehxn/download',
        'ss2015_CBOX_1': 'https://osf.io/g93ef/download',
        'ss2015_CBOX_2': 'https://osf.io/zy853/download',
        'ss2015_CBOX_3': 'https://osf.io/wy9v5/download',
        'ss2016_CBOX_1': 'https://osf.io/2fdgh/download',
        'ss2016_CBOX_2': 'https://osf.io/ngjuf/download',
        'ss2016_CBOX_3': 'https://osf.io/suaty/download',
        'ss2017_CBOX_1': 'https://osf.io/64m9c/download',
        'ss2017_CBOX_2': 'https://osf.io/szgce/download',
        'ss2017_CBOX_3': 'https://osf.io/tg38e/download',
        'ss2018_CBOX_1': 'https://osf.io/ezm3k/download',
        'ss2018_CBOX_2': 'https://osf.io/k8ntg/download',
        'ss2018_CBOX_3': 'https://osf.io/mzv9d/download',
        'ss2019_CBOX_1': 'https://osf.io/46fgp/download',
        'ss2019_CBOX_2': 'https://osf.io/ae83x/download',
        'ss2019_CBOX_3': 'https://osf.io/q296v/download',
        'ss2020_CBOX_1': 'https://osf.io/xsdak/download',
        'ss2020_CBOX_2': 'https://osf.io/tn2pz/download',
        'ss2020_CBOX_3': 'https://osf.io/rwc64/download',
        'ss2021_CBOX_1': 'https://osf.io/tzu37/download',
        'ss2021_CBOX_2': 'https://osf.io/ky4fc/download',
        'ss2021_CBOX_3': 'https://osf.io/vhupa/download',
        'ss2022_CBOX_1': 'https://osf.io/q6nfc/download',
        'ss2022_CBOX_2': 'https://osf.io/k9spw/download',
        'ss2022_CBOX_3': 'https://osf.io/7e693/download',
        'ss2023_CBOX_1': 'https://osf.io/82ntg/download',
        'ss2023_CBOX_2': 'https://osf.io/7zpjc/download',
        'ss2023_CBOX_3': 'https://osf.io/3wspz/download',
        'ss2024_CBOX_1': 'https://osf.io/93cqn/download',
        'ss2024_CBOX_2': 'https://osf.io/npqu9/download',
        'ss2024_CBOX_3': 'https://osf.io/vtmsp/download',
        'ss2025_CBOX_1': 'https://osf.io/4kgde/download',
        'ss2025_CBOX_2': 'https://osf.io/vtu9q/download',
        'ss2025_CBOX_3': 'https://osf.io/ycskx/download',
        'ss2026_CBOX_1': 'https://osf.io/jkxbh/download',
        'ss2026_CBOX_2': 'https://osf.io/6uefm/download',
        'ss2026_CBOX_3': 'https://osf.io/9jskb/download',
        'ss2027_CBOX_1': 'https://osf.io/cyv5z/download',
        'ss2027_CBOX_2': 'https://osf.io/8wzbg/download',
        'ss2027_CBOX_3': 'https://osf.io/zdaen/download',
        'ss2028_CBOX_1': 'https://osf.io/dcjkb/download',
        'ss2028_CBOX_2': 'https://osf.io/7qpf9/download',
        'ss2028_CBOX_3': 'https://osf.io/7jyc6/download',
        'ss2029_CBOX_1': 'https://osf.io/96tn8/download',
        'ss2029_CBOX_2': 'https://osf.io/dfam2/download',
        'ss2029_CBOX_3': 'https://osf.io/k4jbm/download',
        'ss2030_CBOX_1': 'https://osf.io/x9qt3/download',
        'ss2030_CBOX_2': 'https://osf.io/ac4ry/download',
        'ss2030_CBOX_3': 'https://osf.io/ubzsf/download'
        }
    
    response = requests.get(urls_dict[file])
    response.raise_for_status()   

    df = pd.read_csv(io.StringIO(response.content.decode('utf-8'))) 

    # Save the file if a save_path is provided
    if save_path is not None:
        if type == 'csv':
            file_path = os.path.join(save_path, f"{file}.csv")
            df.to_csv(file_path, index=False)
        elif type == 'parquet':
            file_path = os.path.join(save_path, f"{file}.parquet")
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError("Invalid file type. Choose 'csv' or 'parquet'.")
        print(f"File saved as: {file_path}")

    # Convert to parquet format to be returned
    if type == 'parquet':
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        df = pd.read_parquet(parquet_buffer)

    return df


# Example
data2012 = OSF_download('ss2012_CBOX_1', 'csv', "/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git")
data2013 = OSF_download('ss2013_CBOX_1', 'csv', "/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git")
data2014 = OSF_download('ss2014_CBOX_1', 'csv', "/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git")
data2015 = OSF_download('ss2015_CBOX_1', 'csv', "/home/ssk213/CSE_MSE_RXF131/cradle-members/sdle/ssk213/git")
