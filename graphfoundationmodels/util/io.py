""" Auto-Download Module

This function downloads Sunsmart PV Power Plant data from the OSF database. 

"""
import requests
import os

def OSF_download(
    file_number,
    output_file, 
    output_dir
):
    """
    Helper function that downloads CSV files from OSF Database. The urls are already in the function so whichever one can be downloaded

    Args:
        file_number (int): an integer (0-89) that specifies which csv file to download -> 0,1,2 for 2001; 3,4,5 for 2002; etc...
        output_file (string): name for CSV file
        output_dir (string): directory for which the file should be saved in
    """
    
    urls = ['https://osf.io/xcuvg/download', 'https://osf.io/nm9ax/download', 'https://osf.io/zw6as/download',
            'https://osf.io/vu4ym/download', 'https://osf.io/hpgs9/download', 'https://osf.io/b7a6u/download',
            'https://osf.io/emy7z/download', 'https://osf.io/7arj5/download', 'https://osf.io/36pdh/download',
            'https://osf.io/cug32/download', 'https://osf.io/pzv9m/download', 'https://osf.io/gr7cj/download',
            'https://osf.io/kfm2j/download', 'https://osf.io/afdpx/download', 'https://osf.io/afdpx/download',
            'https://osf.io/p3qug/download', 'https://osf.io/tnj8y/download', 'https://osf.io/6wk4x/download',
            'https://osf.io/g49t5/download', 'https://osf.io/tv5xn/download', 'https://osf.io/p2gvq/download',
            'https://osf.io/z29s8/download', 'https://osf.io/7xpsv/download', 'https://osf.io/wg48a/download',
            'https://osf.io/cp4zh/download', 'https://osf.io/qe52w/download', 'https://osf.io/yh3tp/download',
            'https://osf.io/fb73d/download', 'https://osf.io/k3q2u/download', 'https://osf.io/meqs4/download',
            'https://osf.io/x9r2q/download', 'https://osf.io/hafqp/download', 'https://osf.io/c2xj5/download',
            'https://osf.io/tk34n/download', 'https://osf.io/v3xrc/download', 'https://osf.io/xsv3d/download',
            'https://osf.io/937ty/download', 'https://osf.io/j9pes/download', 'https://osf.io/26uts/download',
            'https://osf.io/w6pn8/download', 'https://osf.io/4jxt2/download', 'https://osf.io/fehxn/download',
            'https://osf.io/g93ef/download', 'https://osf.io/zy853/download', 'https://osf.io/wy9v5/download',
            'https://osf.io/2fdgh/download', 'https://osf.io/ngjuf/download', 'https://osf.io/suaty/download',
            'https://osf.io/64m9c/download', 'https://osf.io/szgce/download', 'https://osf.io/tg38e/download',
            'https://osf.io/ezm3k/download', 'https://osf.io/k8ntg/download', 'https://osf.io/mzv9d/download',
            'https://osf.io/46fgp/download', 'https://osf.io/ae83x/download', 'https://osf.io/q296v/download',
            'https://osf.io/xsdak/download', 'https://osf.io/tn2pz/download', 'https://osf.io/rwc64/download',
            'https://osf.io/tzu37/download', 'https://osf.io/ky4fc/download', 'https://osf.io/vhupa/download',
            'https://osf.io/q6nfc/download', 'https://osf.io/k9spw/download', 'https://osf.io/7e693/download',
            'https://osf.io/82ntg/download', 'https://osf.io/7zpjc/download', 'https://osf.io/3wspz/download',
            'https://osf.io/93cqn/download', 'https://osf.io/npqu9/download', 'https://osf.io/vtmsp/download',
            'https://osf.io/4kgde/download', 'https://osf.io/vtu9q/download', 'https://osf.io/ycskx/download',
            'https://osf.io/jkxbh/download', 'https://osf.io/6uefm/download', 'https://osf.io/9jskb/download',
            'https://osf.io/cyv5z/download', 'https://osf.io/8wzbg/download', 'https://osf.io/zdaen/download',
            'https://osf.io/dcjkb/download', 'https://osf.io/7qpf9/download', 'https://osf.io/7jyc6/download',
            'https://osf.io/96tn8/download', 'https://osf.io/dfam2/download', 'https://osf.io/k4jbm/download',
            'https://osf.io/x9qt3/download', 'https://osf.io/ac4ry/download', 'https://osf.io/ubzsf/download']
    
    response = requests.get(urls[file_number])
    response.raise_for_status()  

    file_path = os.path.join(output_dir, output_file)
    with open(file_path, 'wb') as file:
        file.write(response.content)

    print(f"CSV file downloaded as {output_file}")
    return response.status_code

