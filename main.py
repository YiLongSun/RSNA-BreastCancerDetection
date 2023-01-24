import cv2
import dicomsdl
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

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def eval(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    eval_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            pred = model(X)
            eval_loss += loss_fn(pred, Y.unsqueeze(1).float()).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()
    eval_loss /= num_batches
    correct /= size
    print(f"Evaluation Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {eval_loss:>8f}")

def main():
    print(f"Using {DEVICE} device")
    
    # Parameters
    BATCH_TRAIN = 32
    BATCH_EVAL = 16
    EPOCHS = 20
    LR = 0.0005
    
    # Make the training and evaluation csv file
    train_csv = "./Datasets/train.csv"
    train_path = "./Datasets/train_images/"
    target_train = pd.read_csv(train_csv)

    # For Training
    target_path = []
    target_cancer = []
    for index in range(target_train.shape[0]):
        path = train_path \
            +str(target_train["patient_id"][index].item())+"/" \
            +str(target_train["image_id"][index].item())+".dcm"
        cancer = int(target_train["cancer"][index].item())
        target_path.append(path)
        target_cancer.append(cancer)
    target_train_dictionary = {
        "path": target_path,
        "cancer": target_cancer
    }
    
    target_train = pd.DataFrame(target_train_dictionary)
    target_val = target_train.copy()
    target_val = target_val[43766:54707]
    target_train = target_train[0:43766]
    target_train = target_train.reset_index().drop(columns=["index"])
    target_val = target_val.reset_index().drop(columns=["index"])
    
    # Training
    training_data = RSNA_Dataset(target_train)
    val_data = RSNA_Dataset(target_val, is_train = False)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_TRAIN, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_EVAL, shuffle=True)
    
    model = RSNA_Model()
    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        print("\n")
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        eval(val_dataloader, model, loss_fn)
        torch.save(model.state_dict(), f"./Models/model_{epoch}.pth")
    print("Training Done!")

    # Testing
    test_csv = "./Datasets/test.csv"
    test_path = "./Datasets/test_images/"
    target_test = pd.read_csv(test_csv)
    out_predID = []
    out_cancer = []
    index = 0
    for index in range(target_test.shape[0]):
        path = test_path \
            +str(target_test["patient_id"][index].item())+"/" \
            +str(target_test["image_id"][index].item())+".dcm"
        prediction_id = str(target_test["prediction_id"][index])

        dcm_file = dicomsdl.open(str(path))
        image = dcm_file.pixelData()
        image = (image - image.min()) / (image.max() - image.min())
        if dcm_file.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
            image = 1 - image
        image = cv2.resize(image, (224, 224))
        image = (image * 255).astype(np.uint8)
        transform = transforms.Compose([
                transforms.ToTensor()
            ])
        image = transform(image)
        out_image = np.concatenate([image, image, image], axis=0)
        out_image = np.expand_dims(out_image, axis=0)

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