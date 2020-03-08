import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 5           
BATCH_SIZE = 100
LR = 0.001          


# prepare mnist
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),   
    download=False,         
)

# # plot training image
# # plt.imshow(train_data.data[0].numpy(), cmap = 'gray')
# # plt.show()



train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/',
                              train=False,
                              transform=torchvision.transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)


# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[1800:]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels[1800:]




class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 16, 7, 1, 3)
        self.conv2 = nn.Conv2d(16, 16, 7, 1, 3)
        # an affine operation: y = Wx + b
        self.fc = nn.Linear(16 * 28 * 28, 10)  # 6*6 from image dimension
        

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        # If the size is a square you can only specify a single number
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features












def train_and_save():
    cnn = CNN()
    #print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

    # training and testing
    for epoch in range(EPOCH):
        for batch, (b_x, b_y) in enumerate(train_loader): 
            output = cnn(b_x)               # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training batch
            loss.backward()                 # backpropagation, compute gradients
            optimizer.batch()                # apply gradients

    torch.save(cnn, './model/cnn_1(kernel_16_9_16_9_epoch1).pkl')
    print("save complete")



def evaluation():
    cnn = torch.load('./model/cnn_1(kernel_16_9_16_9_epoch1).pkl')
    print("load complete")

    total = 0
    wrong = 0
    for data, target in test_loader:
        #print(target)
        test_output = cnn(data)

        pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

    
        for i in range(pred_y.shape[0]):
            total += 1
            if pred_y[i] != target[i]:
                wrong += 1

    return wrong/total



if __name__ == '__main__':
    #train_and_save()
    loss = evaluation()
    print("loss with 9*9 kernel and epoch=1: ", loss)