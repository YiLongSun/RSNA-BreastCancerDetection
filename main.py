import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import RSNA_Dataset
from model import RSNA_Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(DEVICE), Y.to(DEVICE)

        pred = model(X)
        loss = loss_fn(pred, Y.unsqueeze(1).float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 10 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def eval(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            pred = model(X)
            eval_loss += loss_fn(pred, Y.unsqueeze(1).float()).item()
    eval_loss /= num_batches
    print(f"Evaluation Error: \n Avg loss: {eval_loss:>8f}")

def main():
    print(f"Using {DEVICE} device")
    
    # Parameters
    BATCH_TRAIN = 32
    BATCH_EVAL = 32
    EPOCHS = 100
    LR = 0.00001
    
    model = RSNA_Model()
    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([3.0]).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        print("\n")

        # Make the training and evaluation csv file
        train_csv = "./Datasets/train.csv"
        train_path = "./Datasets/train_images_png_1024/"
        target_train = pd.read_csv(train_csv)
        count_for_normal_patient = 0

        # For Training
        target_path = []
        target_cancer = []
        target_laterality=[]
        for index in range(target_train.shape[0]):
            path = train_path \
                +str(target_train["patient_id"][index].item())+"/" \
                +str(target_train["image_id"][index].item())+".png"
            cancer = int(target_train["cancer"][index].item())
            laterality = target_train["laterality"][index]
            if (cancer == 0 and count_for_normal_patient < 800):
                target_path.append(path)
                target_cancer.append(cancer)
                target_laterality.append(laterality)
                count_for_normal_patient += 1

            if (cancer == 1):
                target_path.append(path)
                target_cancer.append(cancer)
                target_laterality.append(laterality)

        target_train_dictionary = {
            "path": target_path,
            "cancer": target_cancer,
            "laterality": target_laterality
        }
        
        target_train = pd.DataFrame(target_train_dictionary)
        target_val = target_train.copy()
        target_train = target_train.reset_index().drop(columns=["index"])
        target_val = target_val.reset_index().drop(columns=["index"])
        
        # Training
        training_data = RSNA_Dataset(target_train)
        val_data = RSNA_Dataset(target_val, is_train = False)
        train_dataloader = DataLoader(training_data, batch_size=BATCH_TRAIN, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=BATCH_EVAL, shuffle=True)

        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        # eval(val_dataloader, model, loss_fn)
        # torch.save(model.state_dict(), f"./Models/model_{epoch}.pth")
    print("Training Done!")

    # Testing
    test_csv = "./Datasets/test.csv"
    test_path = "./Datasets/test_images_png_1024/"
    target_test = pd.read_csv(test_csv)
    out_predID = []
    out_cancer = []
    index = 0
    for index in range(target_test.shape[0]):
        path = test_path \
            +str(target_test["patient_id"][index].item())+"/" \
            +str(target_test["image_id"][index].item())+".png"
        prediction_id = str(target_test["prediction_id"][index])
        
        image = cv2.imread(path)
        image = cv2.resize(image, (512, 512))
        image = (image * 255).astype(np.uint8)
        transform = transforms.Compose([
                transforms.ToTensor()
            ])
        image = transform(image)
        out_image = np.expand_dims(image, axis=0)

        with torch.no_grad():
            out_image = torch.from_numpy(out_image).float().to(DEVICE)
            pred = model(out_image)
            pred = pred.item()
            print(prediction_id, pred)
            out_predID.append(prediction_id)
            out_cancer.append(pred)
    
    output_dictionary = {
        "prediction_id": out_predID,
        "cancer": out_cancer
    }

    out_csv = pd.DataFrame(output_dictionary)
    out_csv.to_csv('submission.csv', index=False)



if __name__ == "__main__":
    main()