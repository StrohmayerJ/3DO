import torch
import torchvision.models as models
import torch.nn as nn
import datasets as data
import argparse
from tqdm import tqdm

# supress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def train(opt):
    if torch.cuda.is_available():
        device = "cuda:"+opt.device
    else:
        device = "cpu"

    print("Loading 3DO dataset...")
    def load_sequences(day, subfolders):
            return [data.TDODataset(f"{opt.data}/{day}/{subfolder}/csiposreg.csv", opt=opt)for subfolder in subfolders]
    
    # create training and validation dataloader for day 1
    trainSubsets = load_sequences("d1", ["w1", "w2", "w3", "s1", "s2", "s3", "l1", "l2", "l3"]) # ["b1", "b2", "b3"] background class 
    valSubsets = load_sequences("d1", ["w4", "s4", "l4"])
    datasetTrain = torch.utils.data.ConcatDataset(trainSubsets)
    datasetVal = torch.utils.data.ConcatDataset(valSubsets)
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain,batch_size=opt.bs,num_workers=opt.workers,drop_last=True,shuffle=True)
    dataloaderVal = torch.utils.data.DataLoader(datasetVal,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)

    # create test dataloaders for days 1-3 
    testSubsetsD1 = load_sequences("d1", ["w5" ,"s5","l5"])
    testSubsetsD2 = load_sequences("d2", ["w1", "w2", "w3", "w4", "w5", "s1", "s2", "s3", "s4", "s5", "l1", "l2", "l3", "l4", "l5"]) # ["b1", "b2", "b3"] background class 
    testSubsetsD3 = load_sequences("d3", ["w1", "w2", "w4", "w5", "s1", "s2", "s3", "s4", "s5", "l1", "l2", "l4"]) # ["b1", "b2", "b3"] background class
    datasetTestD1 = torch.utils.data.ConcatDataset(testSubsetsD1)
    datasetTestD2 = torch.utils.data.ConcatDataset(testSubsetsD2)
    datasetTestD3 = torch.utils.data.ConcatDataset(testSubsetsD3)
    dataloaderTestD1 = torch.utils.data.DataLoader(datasetTestD1,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)
    dataloaderTestD2 = torch.utils.data.DataLoader(datasetTestD2,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)
    dataloaderTestD3 = torch.utils.data.DataLoader(datasetTestD3,batch_size=opt.bs,num_workers=opt.workers,shuffle=False)

    # create dummy resnet18 model
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # set number of input channels to 1
    model.fc = nn.Linear(512, 3) # set number of output classes to 3
    model.to(device)

    print("Training...")
    for epoch in tqdm(range(opt.epochs), desc='Epochs', unit='epoch'):
        
        # training loop
        for batch in tqdm(dataloaderTrain, desc=f'Epoch {epoch + 1}/{opt.epochs}', unit='batch', leave=False):
            feature_window, l, c = [x.to(device) for x in batch]
            feature_window, l = feature_window.float(), l.float()
            prediction = model(feature_window) # TODO: add your model for training here

        # calidation loop
        with torch.no_grad():
            for batch in tqdm(dataloaderVal):
                feature_window, l, c = [x.to(device) for x in batch]
                feature_window, l = feature_window.float(), l.float()
                prediction = model(feature_window) # TODO: add your model for validation here

    # test loop
    print("Testing...")
    with torch.no_grad():
        for batch in tqdm(dataloaderTestD1):
            feature_window, l, c = [x.to(device) for x in batch]
            feature_window, l = feature_window.float(), l.float()
            prediction = model(feature_window) # TODO: add your model for testing on day 1 here

        for batch in tqdm(dataloaderTestD2):
            feature_window, l, c = [x.to(device) for x in batch]
            feature_window, l = feature_window.float(), l.float()
            prediction = model(feature_window) # TODO: add your model for testing on day 2 here

        for batch in tqdm(dataloaderTestD3):
            feature_window, l, c = [x.to(device) for x in batch]
            feature_window, l = feature_window.float(), l.float()
            prediction = model(feature_window) # TODO: add your model for testing on day 3 here

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/3DO', help='directory of the 3DO dataset')
    parser.add_argument('--ws', type=int, default=351, help='feature window size (i.e. the number of WiFi packets)')
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()

    train(opt)
    print("Done!")
    torch.cuda.empty_cache()


















