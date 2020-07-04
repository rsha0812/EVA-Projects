import torch
import torchvision
import torchvision.transforms as transforms
import Albumentation

def getData():

  k= Albumentation.album_compose()
  transform_test = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
      transforms.RandomRotation((-7.0, 7.0)), transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])
  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=k)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size= 100,
                                            shuffle=True, num_workers=4)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True,transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                          shuffle=False, num_workers=4)

  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return trainloader, testloader, classes