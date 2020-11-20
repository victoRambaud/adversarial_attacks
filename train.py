from __future__ import print_function
import argparse
import torch.utils.data
import torchvision
from torch import optim

from torch.utils.data import DataLoader
from torchvision import transforms

from modules.deep_models import ConvClassifier
from modules.utils import train_epoch, validate_epoch

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 40)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


if __name__ == "__main__":
    dataset = torchvision.datasets.CIFAR10('CIFAR10/train', download=True, train=True,
                                           transform=transforms.ToTensor())
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               lengths=[int(0.8 * len(dataset)),
                                                                        int(0.2 * len(dataset))])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_acc = 0.0
    model = ConvClassifier().to(device)
    output_path = 'models/test'
    model.load_state_dict(torch.load('models/test/last_model.pth'))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.95, 0.999))
    print(device)
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, epoch, train_loader, optimizer, device)
        best_acc = validate_epoch(model, val_loader, output_path, device, best_acc)
