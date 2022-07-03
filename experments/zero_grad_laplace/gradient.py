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
        im_wip.save('resuts/%05d.jpg' % i)
    optimizer.zero_grad()
    x_grad_y = F.conv2d(x, sobel_y).squeeze()
    x_grad_x = F.conv2d(x, sobel_x).squeeze()
    loss = x_grad_x.abs().mean() + x_grad_y.abs().mean()
    loss.backward()
    optimizer.step()
    tbar.set_postfix_str(str(loss.item()))

# x_laplace = F.conv2d(x, laplace).squeeze()
#
# plt.rcParams["figure.figsize"] = (20,10)
#
# plt.subplot(1, 3, 1)
# plt.imshow(x_grad_y, cmap='gray')
# plt.subplot(1, 3, 2)
# plt.imshow(x_grad_x, cmap='gray')
# plt.subplot(1, 3, 3)
# plt.imshow(x_laplace, cmap='gray')
#
# plt.tight_layout()
# plt.show()
