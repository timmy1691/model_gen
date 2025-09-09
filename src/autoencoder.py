import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

# from src.genMLP import genMLP


class mlp(nn.Module):
    def __init__(self, input_dim, output_dim, internal_dims=None, budget=None, optimizer = None):
        super( mlp, self).__init__()
        self.budget = budget
        self.input_dimensions = input_dim
        self.output_dimensions = output_dim
        self.internal_dimensions = internal_dims
        self.model = None

    def initModel(self):
        encoderLayers = []
        if self.budget is None:
            num_dim = 2
            if self.internal_dimensions is None:
                self.internal_dimensions = [int((self.input_dimensions + self.output_dimensions)/2), self.output_dimensions]
            print(self.internal_dimensions)
            for i in range(num_dim):
                if i == num_dim:
                    encoderLayers.append(nn.Linear(self.internal_dimensions[i], self.output_dimensions))
                elif i == 0:
                    encoderLayers.append(nn.Linear(self.input_dimensions, self.internal_dimensions[i]))
                else:
                    encoderLayers.append(nn.Linear(self.internal_dimensions[i-1], self.internal_dimensions[i]))
                encoderLayers.append(nn.ReLU())
        else:
            num_dim = self.budget
            if self.internal_dimensions is None:
                self.internal_dimensions = [int(self.input_dimensions + self.output_dimesnions)/2, self.output_dimensions]
            for i in range(num_dim):
                if i == num_dim:
                    encoderLayers.append(nn.Linear(self.internal_dimensions[i-1], self.output_dimensions))

                elif i == 0:
                    encoderLayers.append(nn.Linear(self.input_dimensions, self.internal_dimensions[i]))
                else:
                    encoderLayers.append(nn.Linear(self.internal_dimensions[i-1], self.internal_dimensions[i]))

                encoderLayers.append(nn.ReLU())

        self.model = nn.Sequential(*encoderLayers)



    def forward(self, X):
        y = self.model(X)
        return y
    

    def trainModel(self, data, learningRate, taskType, epochs=100):
        self.optim = optim.Adam(self.model.parameters(), lr=learningRate)

        if taskType == "regression":
            loss = nn.MSELoss()
        elif taskType == "binary":
            loss = nn.BCELoss()
        else:
            loss = nn.CrossEntropyLoss()
        

        train_data, y_data = data
        for i in range(epochs):
            for j, train_point in enumerate(train_data):
                self.optim.zero_grad()
                y_pred = self.model(train_point)
                l = loss(y_pred, y_data)
                l.backward()
                self.optim.step()

    def testModel(self, testData):
        outputs = []
        
        test_y = self.model(testData)
        outputs.append(test_y.cpu().detach().to_numpy())
        return outputs
    

class AE(nn.Module):
    def __init__(self, input_dim, output_dim, internal_dims=None, budget=None, optimizer = None):
        super(AE, self).__init__()
        self.budget = budget
        self.input_dimensions = input_dim
        self.output_dimensions = output_dim
        self.internal_dimensions = internal_dims
        self.encoder = None
        self.decoder = None
    
    
    def init_model(self):
        self.encoder = mlp(self.input_dimensions, self.output_dimensions, self.internal_dimensions, self.budget)
        self.encoder.initModel()
        if self.internal_dimensions is None:
            self.decoder = mlp(self.output_dimensions, self.input_dimensions, self.internal_dimensions, self.budget)
        else:
            self.decoder = mlp(self.output_dimensions, self.input_dimensions, reversed(self.internal_dimensions), self.budget)
        self.decoder.initModel()
        print("sanity check:" ,self.decoder)

    
    def forward(self, x):
        y = self.encoder(x)
        return y
    
    
    def trainModel(self, train_data, epochs=100, lr=0.05):
        from tqdm import tqdm
        optimizer = optim.SGD(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)
        lossFunction = nn.MSELoss()
        print("start training model")
        for i in tqdm(range(epochs)):
            optimizer.zero_grad()
            if i % 20 == 0:
                acc_loss = 0

            for dp in train_data:
                y_pred = self.decoder(self.encoder((dp)))
                loss = lossFunction(y_pred, dp)
                loss.backward()
                optimizer.step()
                if i % 20 == 0:
                    acc_loss += loss.cpu().detach()

            if i % 20 == 0:
                print(f"The acc loss at epcoh {i}: ", acc_loss)
        print("finished training model")


    def getWeights(self):
        for thing in self.encoder.parameters():
            print(thing)


    def getEmbeddings(self, testData):
        return self.encoder(testData)
    
