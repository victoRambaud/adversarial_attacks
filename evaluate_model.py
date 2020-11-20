import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from modules.utils import accuracy
from modules.deep_models import ConvClassifier


def to_categorical(labels, n=10):
    labels_cat = torch.zeros((len(labels), n), dtype=torch.float32)
    for i, lab in enumerate(labels):
        labels_cat[i, lab] = 1.0
    return labels_cat


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10('CIFAR10/test', download=True, train=False,
                                           transform=transforms.ToTensor())

    classes = dataset.classes
    test_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model_path = 'models/test/best_model.pth'
    model = ConvClassifier()
    model.load_state_dict(torch.load(model_path))

    device = 'cpu'
    model.to(device)
    model.eval()
    test_accuracy = 0.0
    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.to(device)
            preds = model(image)
            test_accuracy += accuracy(preds, label)

        test_accuracy /= len(test_loader.dataset)
    print('test accuracy', 100 * test_accuracy.item())

    # image = image.data.numpy()[0, :, :, :]
    # print('label', classes[label.data.numpy()[0]])
    # print('preds', classes[torch.argmax(preds, axis=1).data.numpy()[0]])
    # print(image.shape)
    # # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # image = (255 * image).astype(np.uint8)
    # print(image)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
