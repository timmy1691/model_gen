import torch 
import torch.nn.functional as f
import torch.nn as nn
from src.helper_function.matGen import genScaledRandMat

class pretrainedCNN(nn.Module):

    # def __init__(self, number_of_kernels, input_dim, kernel_size, input_channels, output_channels, number_of_layers = 2, stride = 1, pad = 1, internal_channels = None):
    def __init__(self, kernel_size, input_channels, output_channels, number_of_layers = 2, stride = 1, pad = 1, internal_channels = None):

        super(pretrainedCNN, self).__init__()
        # self.number_of_kernels = number_of_kernels
        # self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.number_of_layers = number_of_layers
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.pad = pad
        self.internalChannels = internal_channels
        self.model = None

    def initModel(self):
        if self.internalChannels is None:
            num_channels = [self.input_channels]
            for i in range(self.number_of_layers):
                if i == 0:
                    num_channels.append( self.input_channels + (self.output_channels - self.input_channels) // self.number_of_layers)

                if i == self.number_of_layers - 1:
                    num_channels.append(self.output_channels)

            print(num_channels)
            self.internalChannels = num_channels
            
        modelLayers = []
        for i in range(self.number_of_layers):
            modelLayers.append(nn.Conv2d(self.internalChannels[i], self.internalChannels[i+1], self.kernel_size, self.stride, self.pad))
            modelLayers.append(nn.MaxPool2d(2))

        self.model = nn.Sequential(*modelLayers)

    def getWeights(self):
        for name, weights in self.model.named_parameters():
            print(weights)

    def getStateDict(self):
        return self.model.state_dict()


    def genFilters(self):
        if isinstance(self.kernel_size, tuple):
            dimensions = self.kernel_size
        elif isinstance(self.kernel_size, int):
            dimensions = (self.kernel_size, self.kernel_size)
        else:
            raise TypeError("wrong params for the kernel")

        statedict = self.getStateDict()
        newStateDict = statedict.copy()
        for name in statedict:
            if "weight" in name:
                # dim1 = number of out channels
                # dim2 = number of in channels
                # dim3 = size of kernel row
                # dim4 = size of kernel column
                dim1, dim2, dim3, dim4 = statedict[name].shape
                newkernels = []
                for j in range(dim1):
                    # generate a kernel for each out channel
                    num_rows = dim2
                    num_cols = dim3*dim4
                    # generate matrix for input_channel * kernelsize
                    mat = genScaledRandMat(num_rows, num_cols)
                    # print("initialized matrix shape ", mat.shape)
                    newkernels.append(mat.reshape(1, dim2, dim3, dim4))

                newStateDict[name] = torch.concat(newkernels, dim=0)
            
        self.model.load_state_dict(newStateDict)

    def forward(self, x):
        inputs = x
        for module in self.model:
            inputs = module(x)
        
        return inputs


