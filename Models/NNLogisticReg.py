import numpy
import numpy as np

from Models.IModel import IModel
import torch


class NNClassifier(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.Linear(100, 100)
        self.l2 = torch.nn.Linear(100, 4)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, x_train):
        x = self.l1(x_train)
        x = self.relu(x)
        x = self.l2(x)
        x = self.softmax(x)
        return x

    def modify(self, sample: numpy.ndarray):
        l = len(sample)
        self.l1 = torch.nn.Linear(l, 100)


class NNLogisticRegressionModel(IModel):

    def __init__(self):
        self.model: torch.nn = NNClassifier()

    def train(self, x_train, y_train: np.ndarray, *, epoch=1000, batch_size=1024):
        # xtrain is a scipy csr matrix
        sample = x_train[0].toarray()
        self.model.modify(sample)
        self.model.cuda()
        loss_fn = torch.nn.CrossEntropyLoss().cuda()
        optim = torch.optim.Adam(self.model.parameters())
        ptr = 0
        max_size = x_train.shape[0]
        print(x_train.shape)
        print(y_train.shape)

        actual_y = np.zeros((max_size, 4))
        actual_y[y_train] = 1
        actual_y = torch.from_numpy(actual_y).cuda()

        for i in range(epoch):
            if ptr >= max_size:
                ptr = 0
            x_tr = torch.from_numpy(x_train[ptr:max(max_size, ptr + batch_size)].toarray()).cuda()
            y_tr = actual_y[ptr: max(max_size, ptr + batch_size)]
            y_pred = self.model.forward(x_tr)
            loss = loss_fn(y_pred, y_tr)
            loss.backwards()
            optim.step()

    def predict(self, x_test):
        return np.array(self.model.forward(torch.from_numpy(x_test.toarray())))