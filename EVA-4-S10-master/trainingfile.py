from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
train_losses = []
test_losses = []
train_acc = []
test_acc = []


def train(model, device, trainloader, optimizer, epoch):
    model.train()
    pbar = tqdm(trainloader)
    correct = 0
    processed = 0
    train_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)
        train_loss +=loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        train_losses.append(loss)
        train_acc.append(100 * correct / processed)


def test(model, device, testloader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.to(device)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))

    test_acc.append(100. * correct / len(testloader.dataset))
    return test_loss


def Training(epochs, model, device, trainloader, testloader):
    Testloss = 0
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler =  ReduceLROnPlateau(optimizer, 'min') #StepLR(optimizer, step_size=6, gamma=0.1)
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train(model, device, trainloader, optimizer, epoch)
        Testloss = test(model, device, testloader)
        scheduler.step(Testloss)

def plotPerformanceGraph():
  import matplotlib.pyplot as plt
  fig, (axs1,axs2) = plt.subplots(2, 1,figsize=(15,10))

  axs1.plot(train_losses, label = " Train Loss")
  axs1.plot(test_losses, label = " Test Loss")
  axs1.set_title(" Loss")

  axs2.plot(train_acc, label = " Train Accuracy")
  axs2.plot(test_acc, label = " Test Accuracy")

  axs2.set_title(" Accuracy")
  axs1.legend()
  axs2.legend()
  plt.show()


def ClassTestAccuracy(testloader, device, model, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs= outputs.to(device)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))