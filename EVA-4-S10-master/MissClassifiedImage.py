def MissClassifedImage(testloader, model, classes):
  dataiter = iter(testloader)
  import matplotlib.pyplot as plt
  import numpy as np
  import torch
  from GradCam import show_map
  import matplotlib.pyplot as plt

  fig, axs = plt.subplots(25, 3, figsize=(45, 35))
  count = 0
  while True:
      if count >= 25:
          break
      images, labels = dataiter.next()
      output = model(images)
      a, predicted = torch.max(output, 1)
      if (labels != predicted):
          images = images.squeeze()
          heat_map, result = show_map(images, model)
          heat_map = np.transpose(heat_map, (1, 2, 0))
          result = np.transpose(result, (1, 2, 0))
          images = images.cpu()
          images = np.transpose(images, (1, 2, 0))
          axs[count, 0].imshow(images)
          axs[count, 1].imshow(heat_map)
          axs[count, 2].imshow(result)
          axs[count, 0].set_title("Orig: " + str(classes[labels]) + ", Pred: " + str(classes[predicted]))
          fig.tight_layout(pad=3.0)
          count = count + 1
  plt.show()
