# Pytorch
*Train a simple classifier

### Resources
[pytorch](https://pytorch.org/)
[pytorch official tutorial](https://pytorch.org/tutorials/)
[CNN introduciton](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
[CNN stanford](https://cs231n.github.io/convolutional-networks/)

### Homework
 1. update your forked repo from my repo([ref](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork))
 1. following [01_git](../01_git/), create a new branch `LAST#_12pytorch` in your forked repo
 1. activate the environment you created in [03_conda](../03_conda/)
 1. create a notebook under `submissions` and name as `LAST#.ipynb`
 1. start `jupyterlab`, it should open a window in your browser. open your `LAST#.ipynb`
 1. read and understand all sections in pytorch tutorial [Introduction to PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)
 1. load training and validation data use `torchvision.datasets.ImageFolder`, when loading the data, use `torchvision.transforms` to
    - resize the data to 32x32
    - convert the image to `torch.FloatTensor`
    - normalize the imgae by mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
 1. prepare "minibatch" data use `torch.utils.data.DataLoader`
    - set batch_size at 5
    - print the shape of data and label from the training dataloader, you should see 
       ```
       Shape of X [N, C, H, W]: torch.Size([5, 3, 32, 32])
       Shape of y: torch.Size([5]) torch.int64
       ```
1. define a Convolutional Neural Network
    - first convolution: 
       - 2D convolution layer `nn.Conv2d`: set num of out channel at 6, kernel size at 5x5, stride at 1
       - activation function: use ReLU `nn.ReLU` as activation function
       - max pooling layer `nn.MaxPool2d`: set kernel size at 2, stride at 2
    - second convolution: 
       - 2D convolution layer: set num of out channel at 16, kernel size at 5x5, stride at 1
       - activation function: use ReLU as activation function
       - max pooling layer: set kernel size at 2, stride at 2
    - flattening the output and feeding it to a fully connected Neural Network
    - first fully connected layer: 
       - calculate and set the num of in feature
       - set num of out feature at 120
       - activation function: use ReLU `nn.ReLU` as activation function
    - second fully connected layer:
       - set proper num of in feature
       - set num of out feature at 84
       - activation function: use ReLU `nn.ReLU` as activation function
    - last fully connected layer:
       - set proper num of in feature
       - set proper num of out feature, should be num of classes

 1. train a classifier using the defined CNN model:
    - use cross-entropy loss function `nn.CrossEntropyLoss()` and stochastic gradient descent backpropagation `torch.optim.SGD`
    - set learning rate at 0.002
    - train 15 epchos
    - after each epoch, evaluate the model using validation dataloader and calculate the accuracy

 1. stage changes, commit with the message "learning pytorch", push and submit a pr
