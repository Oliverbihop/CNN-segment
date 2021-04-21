from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from torch.utils.data import DataLoader
from dataset import DataLoaderSegmentation
import augmentation as augment
from torch.autograd import Variable
from cbam import *
import aspp
from separableconv import SeparableConv2d


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.leaky1 = nn.LeakyReLU()#negative_slope=0.10000000149011612)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)#,eps=0.01,momentum=0.99)
        self.leaky2 = nn.LeakyReLU()#negative_slope=0.10000000149011612)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1,padding=1)
        self.leaky3 = nn.LeakyReLU()#negative_slope=0.10000000149011612)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1,padding=1)
        self.leaky4 = nn.LeakyReLU()#negative_slope=0.10000000149011612)
        
        # self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        # self.cbam1=CBAM(16,2)

        # self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        # self.cbam2=CBAM(32,2)

        #self.conv5 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        #self.conv6 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        #self.cbam3=CBAM(32,2)

        #self.conv7 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        #self.conv8 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        rates = [1, 3, 6]


        self.aspp1 = aspp.ASPP(128, 64, rate=rates[0])
        self.aspp2 = aspp.ASPP(128, 64, rate=rates[1])
        self.aspp3 = aspp.ASPP(128, 64, rate=rates[2])
        #self.aspp4 = aspp.ASPP(64, 32, rate=rates[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(128, 64, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU())

        self.conv_extra =SeparableConv2d(240+128,128,1)
        self.bn3 = nn.BatchNorm2d(128)

        self.convtran1 = nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0) #20x30

        self.conv9 = nn.Conv2d(128+64, 64, 3, stride=1, padding=1)
        #self.conv10 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.convtran2 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)  #40x60

        self.conv11 = nn.Conv2d(64+32, 32, 3, stride=1, padding=1)
        #self.conv12 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.convtran3 = nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0) #80x120

        self.conv13 = nn.Conv2d(32+16, 16, 3, stride=1, padding=1)
        #self.conv14 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.convtran4 = nn.ConvTranspose2d(16, 16, 2, stride=2, padding=0) #80x120

        self.conv15 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        #self.conv16 = nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.conv17 = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_cat3 = self.leaky1(x)
        x = F.max_pool2d(x_cat3,2)   
        #print(x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        x_cat2 = self.leaky2(x)
        x = F.max_pool2d(x_cat2, 2)
        x = self.conv3(x)
        x_cat1 = self.leaky3(x)
        x = F.max_pool2d(x_cat1, 2)
        x = self.conv4(x)
        x = self.leaky4(x)

        x = self.convtran1(x)
        x = torch.cat((x, x_cat1), dim=1)
        x = self.conv9(x)
        x = F.relu(x)
        #x = self.conv10(x)
        #x = F.relu(x)

        x = self.convtran2(x)
        x = torch.cat((x, x_cat2), dim=1)
        x = self.conv11(x)
        x = F.relu(x)
        # x = self.conv12(x)
        # x = F.relu(x)

        x = self.convtran3(x)
        x = torch.cat((x, x_cat3), dim=1)
        x = self.conv13(x)
        x = F.relu(x)
        # x = self.conv14(x)
        # x = F.relu(x)

        x = self.convtran4(x)
        #x = torch.cat((x, x_cat4), dim=1)
        x = self.conv15(x)
        x = F.relu(x)
        # x = self.conv16(x)
        # x = F.relu(x)

        x = self.conv17(x)
        output = F.sigmoid(x)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(train_loader):
        inputs, labels = sample_batched['image'], sample_batched['label']
                # Forward-Backward of the mini-batch
        inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
        #if CONFIG.USING_GPU:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        output = model(inputs)
        loss = F.binary_cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        
    #mg_size=[160,80]
    composed_transforms = transforms.Compose([
        augment.RandomHorizontalFlip(),
        augment.RandomScale((0.2, .8)),
        augment.RandomCrop(( 160,240)),
        augment.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        augment.ToTensor()])
    lane_train = DataLoaderSegmentation(folder_path='./dataset/',transform=composed_transforms)
    trainloader = DataLoader(lane_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = Net().to(device)
    summary(model,(3,160,240))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        if epoch % 5 == 5 - 1:
            lr_ = lr_poly(args.lr, epoch, args.epochs, 0.9)
            optimizer = optim.Adam(model.parameters(), lr=lr_)
        train(args, model, device, trainloader, optimizer, epoch)
        #test(model, device, test_loader)
        #scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "lane.pt")


if __name__ == '__main__':
    main()
