''' This script transforms the DICOM images to PNG images '''

import os
import dicomsdl
import cv2
import numpy as np

RESIZE = (512, 512)

# Path to the folder containing the DICOM images
TARGET = ""

# Path to the folder where the PNG images will be saved
DESTINATION = ""


def transform_dicom_to_png(target, destination):
    for folder in os.listdir(target):

        for dcm_file in os.listdir(target + folder):

            dcm_path = target + folder + "/" + dcm_file
            dcm = dicomsdl.open(dcm_path)

            image = dcm.pixelData()
            image = (image - image.min()) / (image.max() - image.min())
            if dcm.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
                image = 1 - image
            image = cv2.resize(image, RESIZE)
            image = (image * 255).astype(np.uint8)

            if not os.path.exists(destination + folder):
                os.makedirs(destination + folder)

            cv2.imwrite(destination + folder + "/" + dcm_file[:-4] + ".png", image)

if __name__ == "__main__":
    # transform_dicom_to_png(TARGET, DESTINATION)
    pass