from tqdm import tqdm
import torch


batch_size=64

"""
bceLoss : Reconstruction Loss
mu : Mean
logVar : Log Variance
"""
def lossFunction(bceLoss, mu, logVar):

    #Reconstructional Loss
    BCE = bceLoss

    #KL Divergence Loss
    KLD = -0.5+torch.sum(1+logVar - mu.pow(2) - logVar.exp())

    return BCE+KLD

"""
bceLoss : Reconstruction Loss
mu : Mean
logVar : Log Variance
"""
def train(model, dataloader, dataset, optimizer, criterion):
    model.train()
    runningLoss = 0.0
    counter = 0

    for i,data in tqdm(enumerate(dataloader), total=int(len(dataset)/batch_size)):
        counter += 1
        data = data[0]
        optimizer.zero_grad()

        reconstruction, mu, logVar = model(data)
        bceLoss = criterion(reconstruction, data)

        loss = lossFunction(bceLoss, mu, logVar)
        loss.backward()

        runningLoss += loss.item()

        optimizer.step()

    trainLoss = runningLoss/counter
    return trainLoss

"""
bceLoss : Reconstruction Loss
mu : Mean
logVar : Log Variance
"""
def validate(model, dataloader, dataset, criterion):
    model.eval()
    runningLoss = 0.0

    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/batch_size)):
            counter += 1
            data = data[0]

            reconstruction, mu, logVar = model(data)
            bceLoss = criterion(reconstruction, data)

            loss = lossFunction(bceLoss, mu, logVar)
            runningLoss += loss.item()

            if(i == int(len(dataset)/batch_size)-1):
                reconImage = reconstruction
    valLoss = runningLoss/counter
    return valLoss, reconImage


