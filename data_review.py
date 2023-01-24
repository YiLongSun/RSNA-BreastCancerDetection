import glob
import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DataReviewer():
    def __init__(self, train_csv, train_path):
        self.train_csv = pd.read_csv(train_csv)
        self.train_path = train_path
        
    def show_data_for_patient(self, patient_id):
        patient_dir = os.path.join(self.train_path, str(patient_id))
        num_images = len(glob.glob(f"{patient_dir}/*"))
        print(f"Number of images for patient: {num_images}\n")
        for dcm in os.listdir(patient_dir):
            for index in range(self.train_csv.shape[0]):
                if dcm[0:-4] == str(self.train_csv["image_id"][index]):
                    row = self.train_csv[index:index+1]
            print(f"Patient ID :", row["patient_id"].item())
            # print(f"Site    ID :", row["site_id"].item())
            print(f"Image   ID :", row["image_id"].item())
            # print(f"Machine ID :", row["machine_id"].item())
            print(f"Left/Right :", row["laterality"].item())
            # print(f"View       :", row["view"].item())
            print(f"Age        :", row["age"].item())
            print(f"Cancer     :", row["cancer"].item())
            # print(f"Biopsy     :", row["biopsy"].item())
            # print(f"Invasive   :", row["invasive"].item())
            # print(f"BIRADS     :", row["BIRADS"].item())
            # print(f"Implant    :", row["implant"].item())
            # print(f"Density    :", row["density"].item())
            # print(f"Difficult Negative Case:", row["difficult_negative_case"].item())
            
            dcm_data = pydicom.dcmread(os.path.join(patient_dir, dcm))
            output_data = dcm_data.pixel_array
            if dcm_data.PhotometricInterpretation == "MONOCHROME1":
                output_data = np.amax(output_data) - output_data
            output_data = output_data * dcm_data.RescaleSlope + dcm_data.RescaleIntercept
            plt.imshow(output_data, cmap="bone")
            plt.show()
            print("\n\n")

if __name__ == "__main__":
    train_csv = "./Datasets/train.csv"
    train_path = "./Datasets/train_images/"
    data_reviewer = DataReviewer(train_csv, train_path)
    data_reviewer.show_data_for_patient(10006)