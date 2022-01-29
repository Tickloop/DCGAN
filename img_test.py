PATH = 'data/downsized/1.png'

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dcgan import Generator

chkpt = torch.load("models/10000.pt")

model = Generator()
model.load_state_dict(chkpt["generator"])

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
# optimizer.load_state_dict(chkpt["gOptim"])

# loss = chkpt["loss"]

model.eval()

z = torch.Tensor(np.random.uniform(size=(16, 100)))
image = model(z)
image = image.view(16, 128, 128, 3).detach().numpy()

figure = plt.figure(figsize=(6, 6))

rows = 4
cols = 4
for idx in range(rows * cols):
    figure.add_subplot(rows, cols, idx + 1)
    img = (image[idx] + 1) / 2
    img *= 255
    img = img.astype('int32')
    plt.imshow(img)

plt.show()

