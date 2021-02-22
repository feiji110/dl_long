import torch
from torch.utils.data import DataLoader
from torch import nn,optim
from torchvision import transforms, datasets

from ae import AE

import visdom

def main():

    mnist_train = DataLoader(datasets.MNIST('../Lesson5/mnist_data', True, transform=transforms.Compose([
                    transforms.ToTensor()]),download=True),
                    batch_size=32, shuffle=True)


    mnist_test = DataLoader(datasets.MNIST('../Lesson5/mnist_data', False, transforms.Compose([
                transforms.ToTensor()]),download=True)
                , batch_size=32,shuffle=True)

    x, _ = iter(mnist_train).next()
    print(f'x:{x.shape}')

    device = torch.device('cuda')
    model = AE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    viz = visdom.Visdom()

    for epoch in range(1000):

        for batchidx, (x, _) in enumerate(mnist_train):
            # [b, 1, 28, 28]
            x = x.to(device)

            x_hat,_ = model(x)
            loss = criteon(x_hat, x)


            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        x, _ = iter(mnist_test).next()
        x = x.to(device)
        with torch.no_grad():
            x_hat, kld = model(x)
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))




if __name__ == '__main__':
    main()