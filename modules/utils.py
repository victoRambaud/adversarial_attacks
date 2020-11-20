import torch
import os

from tqdm import tqdm
from torch.nn import functional as F


def accuracy(preds, labels):
    class_preds = torch.argmax(preds, axis=1)
    acc = torch.tensor(torch.sum(class_preds == labels).item(), dtype=torch.float32)
    return acc


def train_epoch(model, epoch, loader, optimizer, device):
    model.train()
    train_loss = 0

    for i, (data, label) in enumerate(tqdm(loader)):
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.cross_entropy(outputs, label, reduction='sum')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(loader.dataset)))


def validate_epoch(model, loader, output_path, device, best_accuracy):
    model.eval()
    test_accuracy = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(tqdm(loader)):
            data = data.to(device)
            labels = labels.to(device)
            preds = model(data)
            test_accuracy += accuracy(preds, labels)

    test_accuracy /= len(loader.dataset)
    torch.save(model.state_dict(), os.path.join(output_path, 'last_model.pth'))
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), os.path.join(output_path, 'best_model.pth'))
        print("Best model saved at :", os.path.join(output_path, 'best_model.pth'))
    print('====> Val set accuracy: {:.4f}'.format(test_accuracy * 100))
    return best_accuracy
