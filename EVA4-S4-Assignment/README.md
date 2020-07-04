# EVA4-S4-Assignment

Model 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        nn.ReLU(),
        nn.BatchNorm2d(16),
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        nn.ReLU(),
        nn.BatchNorm2d(16),
        self.pool1 = nn.MaxPool2d(2, 2)
        nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        nn.ReLU(),
        nn.BatchNorm2d(16),
        self.conv4 = nn.Conv2d(16, 20, 3, padding=1)
        nn.ReLU(),
        nn.BatchNorm2d(20),
        self.pool2 = nn.MaxPool2d(2, 2)
        nn.Dropout(0.2)
        self.conv5 = nn.Conv2d(20, 20, 3)
        nn.ReLU(),
        nn.BatchNorm2d(20),
        self.conv6 = nn.Conv2d(20, 32, 3)
        nn.ReLU(),
        nn.BatchNorm2d(32),
        self.conv7 = nn.Conv2d(32, 10, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)
        
        Epochs Logs:
          0%|          | 0/1875 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:34: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
loss=0.15985998511314392 batch_id=1874: 100%|██████████| 1875/1875 [00:24<00:00, 77.40it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.2825, Accuracy: 8907/10000 (89.07%)

loss=0.37953776121139526 batch_id=1874: 100%|██████████| 1875/1875 [00:23<00:00, 81.48it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.2552, Accuracy: 8978/10000 (89.78%)

loss=0.021813735365867615 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 88.12it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0547, Accuracy: 9845/10000 (98.45%)

loss=0.004770547151565552 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 85.46it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0373, Accuracy: 9886/10000 (98.86%)

loss=0.0003612637519836426 batch_id=1874: 100%|██████████| 1875/1875 [00:20<00:00, 92.32it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0329, Accuracy: 9897/10000 (98.97%)

loss=0.024824947118759155 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.53it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0283, Accuracy: 9909/10000 (99.09%)

loss=4.1812658309936523e-05 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 87.38it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0290, Accuracy: 9906/10000 (99.06%)

loss=0.002127021551132202 batch_id=1874: 100%|██████████| 1875/1875 [00:20<00:00, 89.79it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0297, Accuracy: 9915/10000 (99.15%)

loss=0.15376539528369904 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 86.80it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0289, Accuracy: 9903/10000 (99.03%)

loss=0.00025397539138793945 batch_id=1874: 100%|██████████| 1875/1875 [00:20<00:00, 90.32it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0409, Accuracy: 9891/10000 (98.91%)

loss=0.06868982315063477 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 88.23it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0301, Accuracy: 9911/10000 (99.11%)

loss=0.0021556317806243896 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 87.53it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0346, Accuracy: 9906/10000 (99.06%)

loss=0.0007414519786834717 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 88.21it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0324, Accuracy: 9902/10000 (99.02%)

loss=0.0010007917881011963 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 85.26it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0431, Accuracy: 9904/10000 (99.04%)

loss=0.009555071592330933 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 88.72it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0304, Accuracy: 9919/10000 (99.19%)

loss=0.009514570236206055 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 87.98it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0362, Accuracy: 9910/10000 (99.10%)

loss=7.18832015991211e-05 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 88.34it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0409, Accuracy: 9919/10000 (99.19%)

loss=0.08375835418701172 batch_id=1874: 100%|██████████| 1875/1875 [00:20<00:00, 89.67it/s]
  0%|          | 0/1875 [00:00<?, ?it/s]
Test set: Average loss: 0.0347, Accuracy: 9906/10000 (99.06%)

loss=0.005916237831115723 batch_id=1874: 100%|██████████| 1875/1875 [00:21<00:00, 87.33it/s]

Test set: Average loss: 0.0370, Accuracy: 9894/10000 (98.94%)
