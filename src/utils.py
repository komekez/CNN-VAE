import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

toPilImage = transforms.ToPILImage()


def imageToVid(images):
    imgs = [np.array(toPilImage(img)) for img in images]
    imageio.mimsave('generted_images.gif', imgs)

def saveReconstructedImage(reconImage, epoch):
    save_image(reconImage, f'output{epoch}.jpg')

def saveLossPlot(trainLoss, validLoss):
    plt.figure(figsize=(10, 7))
    plt.plot(trainLoss, color='orange', label='train loss')
    plt.plot(validLoss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.jpg')
    plt.show()

