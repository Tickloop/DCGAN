from asyncore import write
from typing import Dict
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running model on device: ", device)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="downsized", type=str, dest="data_dir", help="Whether to use simplified or cropped images")
parser.add_argument('--channels', default=3, type=int, help="The number of channels to store in the image. 0 will store image in gray, 3 for rgb/bgr")
parser.add_argument('--batch_size', default=100, type=int, help="The number of samples to be used in one forward pass of generator and discriminator")
parser.add_argument('--epochs', default=100, type=int, help="Number of iterrations to train the network for")
parser.add_argument('--latent_dim', default=100, type=int, help="The dimenssion of noise to be fed into the generator network")
parser.add_argument('--no_samples', default=True, action='store_false', dest="samples", help="Whether to store samples from each epoch or not")
options = parser.parse_args()

print("Arguments for current run: ")
print(options)

img_size = 128 if options.data_dir == "downsized" else 200

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.channels = [243, 81, 27, 9, 3]
        self.init_size = 8

        # mapping our latent dimenssion to something that can be turned into something convolutable
        self.linear = nn.Linear(options.latent_dim, self.channels[0] * self.init_size ** 2)
        
        def ConvBlock(inChannels, outChannels, last=False):
            layers = [
                nn.ConvTranspose2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(outChannels),
                nn.Dropout(p=0.2)
            ]

            if last:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            
            return layers


        # need to reshape the output to N x C x H x W before passing into model
        # * operator is used to unpack the list returned from ConvBlock
        self.model = nn.Sequential(
            nn.BatchNorm2d(self.channels[0]),
            *ConvBlock(self.channels[0], self.channels[1]),
            *ConvBlock(self.channels[1], self.channels[2]),
            *ConvBlock(self.channels[2], self.channels[3]),
            *ConvBlock(self.channels[3], self.channels[4], last=True)
        )
    
    def forward(self, x):
        img = self.linear(x)
        # reshape img to be of the shape (N x C x H x W): (Batchsize, Number of channels, ImgHeight, ImgWidth)
        img = img.view(img.shape[0], self.channels[0], self.init_size, self.init_size)
        img = self.model(img)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.channels = [3, 9, 27, 81, 243]
        self.final_size = 8
        
        def ConvBlock(inChannels, outChannels):
            return [
                nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(outChannels),
                nn.LeakyReLU(0.2),
            ]
        
        self.model = nn.Sequential(
            *ConvBlock(self.channels[0], self.channels[1]),
            *ConvBlock(self.channels[1], self.channels[2]),
            *ConvBlock(self.channels[2], self.channels[3]),
            *ConvBlock(self.channels[3], self.channels[4])
        )


        # need to reshape to be of the form (N, C * H * W)
        self.linear = nn.Sequential(
            nn.Linear(self.channels[4] * self.final_size ** 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        is_valid = self.model(x)
        # reshape to fit the linear layer of size (N, C * H * W)
        is_valid = is_valid.view(is_valid.shape[0], -1)
        is_valid = self.linear(is_valid)
        return is_valid

def dataLoader(images):
    """ 
        Loads the dataset into an array that has images from https://www.kaggle.com/kostastokis/simpsons-faces 
        If simplified is passed as a flag then simplified images are used
        If channels is 0 then gray images are used
    """
    COLOR = cv2.IMREAD_COLOR if options.channels == 3 else cv2.IMREAD_GRAYSCALE
    for path, dirs, files in os.walk(f"data/{options.data_dir}"):
        for file in tqdm(files):
            img = cv2.imread(f"{path}/{file}", COLOR)
            images.append(img)
    rows = 4
    cols = 4
    figure = plt.figure(figsize=(6, 6))
    choices = np.random.choice(np.arange(len(images)), rows * cols)
    
    for idx in range(rows * cols):
        figure.add_subplot(rows, cols, idx + 1)
        choice = choices[idx]
        img = cv2.cvtColor(images[choice], cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    
    plt.show()
    plt.close(figure)


def train(generator, discriminator, images):
    # optimizers that we will use
    generater_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # the loss function
    adv_loss = nn.BCELoss()

    # total number of real data points that we have
    N = images.shape[0]

    gLoss = np.zeros(options.epochs)
    dLoss = np.zeros(options.epochs)

    for epoch in range(options.epochs):
        real_images = np.array([images[choice] for choice in np.random.choice(np.arange(N), size=options.batch_size)])

        # rescale values between [-1 and 1]
        real_images = 2.0 * (real_images - np.min(real_images)) / np.ptp(real_images) - 1
        real_images = torch.Tensor(real_images).to(device)
        # fancy way to reshape the images to be of the shape (N, C, H, W)
        real_images = real_images.view(real_images.shape[0], *real_images.shape[1:][::-1])
        real_labels = torch.Tensor(np.ones((options.batch_size, 1))).to(device)
        
        z = torch.Tensor(np.random.uniform(size=(options.batch_size, options.latent_dim))).to(device)
        
        fake_images = generator(z)
        fake_labels = torch.Tensor(np.zeros((options.batch_size, 1))).to(device)

        # discriminator training
        discriminator_optimizer.zero_grad()

        real_loss = adv_loss(discriminator(real_images), real_labels)
        fake_loss = adv_loss(discriminator(fake_images), fake_labels)
        discriminator_loss = (real_loss + fake_loss) / 2
        dLoss[epoch] = discriminator_loss.item()

        # back prop
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # generator training
        generater_optimizer.zero_grad()

        z = torch.Tensor(np.random.uniform(size=(options.batch_size, options.latent_dim))).to(device)
        fake_images = generator(z)

        generator_loss = adv_loss(discriminator(fake_images), real_labels)
        gLoss[epoch] = generator_loss.item()

        # backprop
        generator_loss.backward()
        generater_optimizer.step()

        print(f"Epoch {epoch + 1} / {options.epochs}: Generator Loss: {generator_loss} Discriminator Loss: {discriminator_loss}")

        writeImages(fake_images, epoch)
    
    plotLoss(gLoss, dLoss)
    
    return discriminator_optimizer, generater_optimizer, adv_loss

def writeImages(batch, epoch):
    """
        Function to make our output for each epoch more meaningful as well as better represented
    """
    if options.samples:
        image = batch[0].cpu().detach()
        image = image.view(img_size, img_size, options.channels).numpy()
        image = (image + 1) / 2
        image *= 255

        # writing this for sanity check
        cv2.imwrite(f"./samples/sample_epoch_{epoch}.jpg", image)

def save_models(generator, discriminator, **kwargs):
    to_save = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }

    if "loss" in kwargs:
        to_save["loss"] = kwargs.get("loss")
    
    if "gOptim" in kwargs:
        to_save["gOptim"] = kwargs.get("gOptim").state_dict()
    
    if "dOptim" in kwargs:
        to_save["dOptim"] = kwargs.get("dOptim").state_dict()
    
    torch.save(to_save, f"models/{options.epochs}.pt")

def plotLoss(gLoss, dLoss):
    x = np.arange(0, options.epochs, 1)
    plt.plot(x, gLoss, 'r', x, dLoss, 'g')
    plt.savefig(f"plots/loss_plot_{options.epochs}.png")

def main():
    images = []

    dataLoader(images)
    images = np.array(images)

    print("Images loaded", images.shape)

    generator = Generator()
    discriminator = Discriminator()
    generator.to(device)
    discriminator.to(device)

    g_optimizer, d_optimizer, loss_func = train(generator, discriminator, images)

    save_models(generator, discriminator, loss=loss_func, gOptim=g_optimizer, dOptim=d_optimizer)

if __name__ == "__main__":
    main()
