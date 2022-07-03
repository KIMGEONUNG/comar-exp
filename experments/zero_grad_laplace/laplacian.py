from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor, Compose, Grayscale
import matplotlib.pyplot as plt
from tqdm import tqdm


sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])\
        .float()\
        .unsqueeze(0)\
        .unsqueeze(0)

sobel_x = sobel_y.permute(0, 1, 3, 2)

laplace = torch.Tensor([[0, -1, 0], [-1, 4 ,-1], [0, -1, 0]])\
        .float()\
        .unsqueeze(0)\
        .unsqueeze(0)

im = Image.open('./srcs/imgs/sample01.jpg')

prep = Compose([ToTensor(), Grayscale()])
x = prep(im).unsqueeze(0)
x = Variable(x, requires_grad=True)

# Define optimizer
optimizer = optim.Adam([x], 0.001)

num_iter = 2000
tbar = tqdm(range(num_iter))
for i in tbar:
    if i % 10 == 0:
        im_wip = ToPILImage()(x.detach().squeeze())
        im_wip.save('results/%05d.jpg' % i)
    optimizer.zero_grad()
    x_laplace = F.conv2d(x, laplace).squeeze()
    loss = x_laplace.abs().mean()
    loss.backward()
    optimizer.step()
    tbar.set_postfix_str(str(loss.item()))
