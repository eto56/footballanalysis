
# process npz files


import numpy as np
import os
import cv2


import glob

keys = ["K", "R", "t" , "k"]

def process_npz(npz_path, output_path):
    """
    Process npz files and save as images.

    Parameters
    ----------
    npz_path : str
        Path to the npz file.
    output_path : str
        Path to the output directory.
    """
    print (npz_path)
    npz = np.load(npz_path)
    #print (npz)
    for key in npz.keys():
        print (key)

    return

    for i in range(len(npz['frames'])):
        frame = npz['frames'][i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_path, f'{i:04d}.png'), frame)

#/home/eto/projects/clones/footballanalysis/worldpose/_cameras-dev/cameras/ARG_CRO_220001.npz

if __name__ == '__main__':
    data_path = "_cameras-dev/cameras/*.npz"
    output_path = "output/"
    os.makedirs(output_path, exist_ok=True)
    for npz_path in glob.glob(data_path):
        process_npz(npz_path, output_path)
