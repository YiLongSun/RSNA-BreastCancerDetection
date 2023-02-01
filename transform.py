# Description: This script transforms the DICOM images to PNG images
import os
import dicomsdl
import cv2
import numpy as np

target_dir = "./train_images/"
target_destination = "./train_images_png/"

def transform_dicom_to_png(target_dir, target_destination):
    for folder in os.listdir(target_dir):
        for folder in os.listdir(target_dir):
            for dcm_file in os.listdir(target_dir + folder):
                dcm_path = target_dir + folder + "/" + dcm_file
                dcm = dicomsdl.open(dcm_path)
                image = dcm.pixelData()
                image = (image - image.min()) / (image.max() - image.min())
                if dcm.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
                    image = 1 - image
                image = cv2.resize(image, (224, 224))
                image = (image * 255).astype(np.uint8)

                if not os.path.exists(target_destination + folder):
                    os.makedirs(target_destination + folder)

                cv2.imwrite(target_destination + folder + "/" + dcm_file[:-4] + ".png", image)

if __name__ == "__main__":
    transform_dicom_to_png(target_dir, target_destination)