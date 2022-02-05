import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running model on device: ", device)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="simpsons_simplified_64", type=str, help="Whether to use simplified or cropped images")
parser.add_argument('--batch_size', default=128, type=int, help="The number of samples to be used in one forward pass of generator and discriminator")
parser.add_argument('--epochs', default=100, type=int, help="Number of iterrations to train the network for")
parser.add_argument('--no_gif', default=True, action='store_false', dest="gif", help="Whether to store the gif created from samples or not")
parser.add_argument('--no_preview', default=True, action='store_false', dest="preview", help="Whether to show a preview of the loaded dataset or not")
parser.add_argument('--no_samples', default=True, action='store_false', dest="samples", help="Whether to store a sample from each epoch or not")
parser.add_argument('--figure_every', default=100, type=int, help="The number of epochs after which a figure of 16 random images from a batch are tested")
parser.add_argument('--checkpoint_every', default=1000, type=int, help="The number of epochs after which generator and discriminator model are saved")
parser.add_argument('--fps', default=60, type=int, help="The frames per second for creating a gif")
options = parser.parse_args()

print("Arguments for current run: ")
print(options)

if options.data_dir == "simpsons_128":
    img_size = 128
elif options.data_dir == "celeba_64" or options.data_dir == "simpsons_simplified_64":
    img_size = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.channels = [1024, 512, 256, 128, 64, 3]
        self.init_size = img_size // ( 2 ** (len(self.channels) - 1))

        # mapping our latent dimenssion to something that can be turned into something convolutable
        self.linear = nn.Sequential(
            nn.Linear(options.latent_dim, self.channels[0] * self.init_size ** 2),
            nn.ReLU()
        )
        
        def ConvBlock(inChannels, outChannels, last=False):
            layers = [
                nn.ConvTranspose2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1),
            ]

            if last:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.BatchNorm2d(outChannels))
                layers.append(nn.ReLU())
            
            return layers


        # need to reshape the output to N x C x H x W before passing into model
        # * operator is used to unpack the list returned from ConvBlock
        self.model = nn.Sequential(
            nn.BatchNorm2d(self.channels[0]),
            *ConvBlock(self.channels[0], self.channels[1]),
            *ConvBlock(self.channels[1], self.channels[2]),
            *ConvBlock(self.channels[2], self.channels[3]),
            *ConvBlock(self.channels[3], self.channels[4]),
            *ConvBlock(self.channels[4], self.channels[5], last=True)
        )

        self.linear.apply(initWeights)
    
    def forward(self, x):
        img = self.linear(x)
        # reshape img to be of the shape (N x C x H x W): (Batchsize, Number of channels, ImgHeight, ImgWidth)
        img = img.view(img.shape[0], self.channels[0], self.init_size, self.init_size)
        img = self.model(img)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.channels = [3, 64, 128, 256, 512, 1024]
        self.final_size = img_size // ( 2 ** (len(self.channels) - 1))
        
        def ConvBlock(inChannels, outChannels, last=False):
            if last:
                return [
                    nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                ]
            else:
                return [
                    nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(outChannels),
                    nn.LeakyReLU(0.2),
                ]
        
        self.model = nn.Sequential(
            *ConvBlock(self.channels[0], self.channels[1]),
            *ConvBlock(self.channels[1], self.channels[2]),
            *ConvBlock(self.channels[2], self.channels[3]),
            *ConvBlock(self.channels[3], self.channels[4]),
            *ConvBlock(self.channels[4], self.channels[5], last=True)
        )

        # need to reshape to be of the form (N, C * H * W)
        self.linear = nn.Sequential(
            nn.Linear(self.channels[-1] * self.final_size ** 2, 1),
            nn.Sigmoid(),
        )
        
        self.linear.apply(initWeights)

    def forward(self, x):
        is_valid = self.model(x)
        # reshape to fit the linear layer of size (N, C * H * W)
        is_valid = is_valid.view(is_valid.shape[0], -1)
        is_valid = self.linear(is_valid)
        return is_valid

def showPreview(images):
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

def dataLoader(images):
    """ 
        Loads the dataset into an array that has images from https://www.kaggle.com/kostastokis/simpsons-faces 
        If simplified is passed as a flag then simplified images are used
        If channels is 0 then gray images are used
    """
    for path, dirs, files in os.walk(f"data/{options.data_dir}"):
        for file in tqdm(files):
            img = cv2.imread(f"{path}/{file}")
            images.append(img)
    
    options.preview and showPreview(images)
    

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

    """ In order to maintain our samples, we will create a fixed input vector to get 16 samples """
    z_fixed = torch.Tensor(np.random.uniform(size=(1, options.latent_dim))).to(device)

    for epoch in range(options.epochs):
        real_images = np.array([images[choice] for choice in np.random.choice(np.arange(N), size=options.batch_size)])

        # rescale values between [-1 and 1] that is the output of our tanh function
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

        fake_image = generator(z_fixed)
        options.samples and writeSample(fake_image, epoch)

        if (epoch + 1) % options.sample_every == 0:
            print(f"WRITIGN SAMPLES - {epoch}/{options.epochs}")
            writeFigures(fake_images, epoch)
        
        if (epoch + 1) % options.checkpoint_every == 0:
            print(f"SAVING CHECKPOINT - {epoch}/{options.epochs}")
            saveModels(generator, discriminator, epoch)
    
    plotLoss(gLoss, dLoss)
    
    return discriminator_optimizer, generater_optimizer, adv_loss

def writeSample(batch, epoch):
    """
        Function to make our output for each epoch more meaningful as well as better represented
    """
    if options.samples:
        image = batch[0].cpu().detach()
        image = image.view(img_size, img_size, options.channels).numpy()
        image = (image + 1) / 2
        image *= 255

        # writing this for sanity check
        cv2.imwrite(f"samples/{options.run_name}/sample_epoch_{epoch}_of_{options.epochs}.jpg", image)

def saveModels(generator, discriminator, epoch=None):
    generator_model = torch.jit.script(generator)
    if epoch != None:
        generator_model.save(f"models/{options.run_name}/checkpoints/{epoch}_of_{options.epochs}_{options.batch_size}_{img_size}_{options.data_dir}_g.pt")
    else:
        generator_model.save(f"models/{options.run_name}/{options.epochs}_{options.batch_size}_{img_size}_{options.data_dir}_g.pt")

    discriminator_model = torch.jit.script(discriminator)
    if epoch != None:
        discriminator_model.save(f"models/{options.run_name}/checkpoints/{epoch}_of_{options.epochs}_{options.batch_size}_{img_size}_{options.data_dir}_d.pt")
    else:
        discriminator_model.save(f"models/{options.run_name}/{options.epochs}_{options.batch_size}_{img_size}_{options.data_dir}_d.pt")

def plotLoss(gLoss, dLoss):
    x = np.arange(0, options.epochs, 1)
    plt.plot(x, gLoss, 'r', x, dLoss, 'g')
    plt.savefig(f"plots/{options.run_name}/loss_plot_{options.epochs}_{options.batch_size}_{img_size}_{options.data_dir}.png")
    plt.close('all')

def writeFigures(images, epoch):
    images = images.cpu().detach()
    choices = np.random.choice(np.arange(options.batch_size), size=16)
    rows = 4
    cols = 4
    figure = plt.figure(figsize=(6, 6))

    for idx in range(rows * cols):
        figure.add_subplot(rows, cols, idx + 1)
        img = images[choices[idx]]
        img = img.view(img_size, img_size, 3).numpy()
        img = (img + 1) / 2
        img *= 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    
    plt.savefig(f"figures/{options.run_name}/{epoch}_of_{options.epochs}_{img_size}_{options.data_dir}.png")
    plt.close(figure)

def makeDirs():
    run_name = f"{options.data_dir}/run_{options.epochs}_{options.batch_size}"
    parent_dirs = ["samples", "plots", "models", "gifs", "figures"]

    # directory for samples, plots, models, gifs, figures corresponding to this run
    for dir in parent_dirs:
        if not os.path.isdir(f"{dir}/{run_name}"):
            os.makedirs(f"{dir}/{run_name}")
    
    if not os.path.isdir(f"models/{run_name}/checkpoints"):
        os.mkdir(f"models/{run_name}/checkpoints")

    options.run_name = run_name

def makeGif():
    if not options.no_gif:
        print("Making gif of samples")
        gif = []

        for filename in tqdm(range(options.epochs)):
                gif.append(imageio.imread(f"samples/{options.run_name}/sample_epoch_{filename}_of_{options.epochs}.jpg"))

        imageio.mimsave(f"gifs/{options.run_name}/{options.epochs}_{options.batch_size}.gif", gif, fps=options.fps)

def initWeights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.02)

def main():
    images = []

    dataLoader(images)
    images = np.array(images)

    print("Images loaded", images.shape)

    makeDirs()

    generator = Generator()
    discriminator = Discriminator()
    generator.to(device)
    discriminator.to(device)

    train(generator, discriminator, images)

    saveModels(generator, discriminator)

    options.gif and makeGif()

if __name__ == "__main__":
    main()
