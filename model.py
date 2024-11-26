import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


def get_model(name, params):
    models = {'CRNN_v4':CRNN_v4(params),
              'CRNN_v4a':CRNN_v4a(params),
              'CRNN_v4b':CRNN_v4b(params)}
    
    model = models.get(name)
    if model is None:
        raise ValueError(f'Model {name} not impelemted')
    return model


class CRNN_v4(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN_v4, self).__init__()
        resnet = tv.models.resnet18(weights=tv.models.resnet.ResNet18_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-3]))
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, vocab_size)
        self.gru1 = nn.GRU(input_size=1024, hidden_size=1024)

    def forward(self, x):
        x = self.resnet(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        x = F.dropout(self.fc1(x), p=0.5)
        output, _ = self.gru1(x)
        x = self.fc2(output)
        x = x.permute(1, 0, 2)
        return x
    

class CRNN_v4a(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN_v4a, self).__init__()
        resnet = tv.models.resnet18(weights=tv.models.resnet.ResNet18_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-3]))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 512)
        self.gru1 = nn.GRU(input_size=512, hidden_size=512, bidirectional=True)
        self.fc2 = nn.Linear(1024, vocab_size)

    def forward(self, x):
        x = self.resnet(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        x = F.dropout(self.fc1(x), p=0.5)
        x = self.relu(x)
        output, _ = self.gru1(x)
        x = self.fc2(output)
        x = x.permute(1, 0, 2)
        return x


class CRNN_v4b(nn.Module):
    def __init__(self, vocab_size):
        super(CRNN_v4b, self).__init__()
        resnet = tv.models.resnet18(weights=tv.models.resnet.ResNet18_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-3]))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 512)
        self.gru1 = nn.GRU(input_size=512, hidden_size=512, bidirectional=True, num_layers=3)
        self.fc2 = nn.Linear(1024, vocab_size)

    def forward(self, x):
        x = self.resnet(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        x = F.dropout(self.fc1(x), p=0.5)
        x = self.relu(x)
        output, _ = self.gru1(x)
        x = self.fc2(output)
        x = x.permute(1, 0, 2)
        return x

