import torch
import torch.nn as nn
import torch.nn.functional as F


#DeepEC Model
class PlasticModel(nn.Module):
    def __init__(self, kernel_size):
        super(PlasticModel, self).__init__()
        
        eps_value = 1e-01
        momentum_value = 0.99
        max_kernel_size = 600 + 1#seqeunce length + 1 - filter_size
        
        self.kernel_size = kernel_size
        
        conv1 = nn.Conv2d(1, 128, kernel_size=(4, self.kernel_size), stride=1)
        pool1 = nn.MaxPool2d(kernel_size=(max_kernel_size - 4, 1))

        conv2 = nn.Conv2d(1, 128, kernel_size=(8, self.kernel_size), stride=1)
        pool2 = nn.MaxPool2d(kernel_size=(max_kernel_size - 8, 1))

        conv3 = nn.Conv2d(1, 128, kernel_size=(16, self.kernel_size), stride=1)
        pool3 = nn.MaxPool2d(kernel_size=(max_kernel_size - 16, 1))
        
        batch1 = nn.BatchNorm2d(128, momentum = momentum_value, eps = eps_value)
        
        conv1 = self.init_layer(conv1)
        conv2 = self.init_layer(conv2)
        conv3 = self.init_layer(conv3)
        
        self.conv1_module = nn.Sequential(conv1, batch1, nn.ReLU(), pool1)
        self.conv2_module = nn.Sequential(conv2, batch1, nn.ReLU(), pool2)
        self.conv3_module = nn.Sequential(conv3, batch1, nn.ReLU(), pool3)
        
    def forward(self, x):
        out1 = self.conv1_module(x)
        out2 = self.conv2_module(x)
        out3 = self.conv3_module(x)

        return out1, out2, out3

    def init_layer(self, layer):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0)
        return layer

#BRNN
class BRNN(nn.Module):
    
    def __init__(self, in_size):
        super(BRNN, self).__init__()

        self.in_size = in_size
        self.num_layers = 2
        self.hidden_size = 256
        self.lstm1 = nn.LSTM(input_size = self.in_size, hidden_size = self.hidden_size, num_layers = self.num_layers, bidirectional = True)
        self.lstm2 = nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 2, bidirectional = True)

        self.in_linear = nn.Linear(256000, 1024)
        self.mid_linear = nn.Linear(1024, 512)
        self.hd_linear = nn.Linear(512, 14)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        
        # fc1 = nn.Linear(384, 512)
        # fc2 = nn.Linear(512, 512)
        # fc3 = nn.Linear(512, 4) # 4 는 4class 전용
        
        self.lstm1 = self.init_lstm_bias_weight(self.lstm1)
        self.lstm2 = self.init_lstm_bias_weight(self.lstm2)
        
        self.in_linear = self.init_layer(self.in_linear)
        self.mid_linear = self.init_layer(self.mid_linear)
        self.hd_linear = self.init_layer(self.hd_linear)
        
        self.fc_module = nn.Sequential(self.in_linear, self.relu,
                                       self.mid_linear, self.relu, self.hd_linear)
    def init_layer(self, layer):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0)
        return layer
        
    def forward(self, data):
        data = data[:, :500, :]
        h0 = torch.zeros(self.num_layers*2, data.size(1), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, data.size(1), self.hidden_size).cuda()

        out, h_out = self.lstm1(data, (h0, c0))
        out = self.relu(out)
        out, _ = self.lstm2(out, h_out)
        out = self.relu(out)
        
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return out
        # return self.sigmoid(out)
        
    def init_lstm_bias_weight(self, layer):
        nn.init.xavier_normal_(layer.weight_hh_l0)
        nn.init.xavier_normal_(layer.weight_hh_l0_reverse)
        nn.init.xavier_normal_(layer.weight_ih_l0)
        nn.init.xavier_normal_(layer.weight_ih_l0_reverse)
        nn.init.xavier_normal_(layer.weight_hh_l1)
        nn.init.xavier_normal_(layer.weight_hh_l1_reverse)
        nn.init.xavier_normal_(layer.weight_ih_l1)
        nn.init.xavier_normal_(layer.weight_ih_l1_reverse)

        layer.bias_hh_l0.data.fill_(0)
        layer.bias_hh_l0_reverse.data.fill_(0)
        layer.bias_ih_l0.data.fill_(0)
        layer.bias_ih_l0_reverse.data.fill_(0)
        layer.bias_hh_l1.data.fill_(0)
        layer.bias_hh_l1_reverse.data.fill_(0)
        layer.bias_ih_l1.data.fill_(0)
        layer.bias_ih_l1_reverse.data.fill_(0)

        return layer