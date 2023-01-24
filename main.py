import cv2
import dicomsdl
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

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
            
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    print(f"Using {DEVICE} device")
    
    # Parameters
    BATCH_TRAIN = 64
    BATCH_TEST = 32
    EPOCHS = 3
    LR = 0.0005
    
    # Make the training csv file
    train_csv = "./Datasets/train.csv"
    train_path = "./Datasets/train_images/"
    target_csv = pd.read_csv(train_csv)
    target_path = []
    target_cancer = []
    
    for index in range(target_csv.shape[0]):
        path = train_path \
            +str(target_csv["patient_id"][index].item())+"/" \
            +str(target_csv["image_id"][index].item())+".dcm"
        cancer = int(target_csv["cancer"][index].item())
        target_path.append(path)
        target_cancer.append(cancer)
        
    target_dictionary = {
        "path": target_path,
        "cancer": target_cancer
    }
    
    target_csv = pd.DataFrame(target_dictionary)
    
    # Training
    training_data = RSNA_Dataset(target_csv)
    testing_data = RSNA_Dataset(target_csv, is_train = False)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_TRAIN, shuffle=True)
    test_dataloader = DataLoader(training_data, batch_size=BATCH_TEST, shuffle=True)
    
    model = RSNA_Model()
    model = model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    main()