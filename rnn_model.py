import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer, batch_first, batch_size):
        super(LSTM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, 1), padding=0)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first=batch_first)
        # self.lstm2 = nn.LSTM(hidden_size, 32, num_layer, batch_first=batch_first)
        # self.lstm3 = nn.LSTM(32, 16, num_layer, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, 2)
        # self.fc2 = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net(self, x):  # x:[batch, seq_len, hidden_dim]
        # 原文中使用了 dropout 此处弃用 https://www.cnblogs.com/cxq1126/p/13504437.html
        u = torch.tanh(torch.matmul(x, self.w_omega))  # [batch, seq_len, hidden_dim]
        att = torch.matmul(u, self.u_omega)  # [batch, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score  # [batch, seq_len, hidden_dim]
        context = torch.sum(scored_x, dim=1)  # [batch, hidden_dim]
        return context

    def forward(self, x, length_list):
        # 解决同一 batch 数据变长问题
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=length_list, batch_first=True)
        x, (hn, cn) = self.lstm(x)
        # x, (hn, cn) = self.lstm2(x)
        # x, (hn, cn) = self.lstm3(x)
        x, y = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # x = x.permute(1, 0, 2)  # [batch, seq_len, hidden_dim]
        # attn_output = self.attention_net(x)
        # s = torch.sigmoid(attn_output)
        # logit = self.fc(attn_output)
        # torch.sigmoid(torch.sum(attn_output, dim=1))
        # t = self.fc(x)
        # self.fc(x).squeeze()
        # self.fc(x[-1]).view(-1)
        # x = x.transpose(0, 1)
        #  fc1 = F.relu(self.fc(x))  # torch.Size([1, 32, 5])
        # output = self.fc2(fc1)  # torch.Size([1, 32, 2])
        # output = F.softmax(output, dim=2)  # torch.Size([1, 32, 2])       (概率表示：行加起来为1)
        # output = output.transpose(0, 1).contiguous()
        # x = x[:, -1, :]  # 取 ht 时刻的输出
        # x = self.fc(hn[-1])
        # x = self.fc2(x)
        # x = self.attention_net(x)
        # x = self.fc(x)
        # x = self.fc2(x)
        return self.fc(hn[-1])
        # return self.sigmoid(x).squeeze(dim=1)
        # x = x.unsqueeze(1)  # [batch,input_channel,high,width]
        # x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = x.squeeze(dim=1)
        # h1,h2,h3 = x.size()
        # x, (hn, cn) = self.lstm(x)
        # x = self.fc(hn[-1])
        # return self.sigmoid(x).squeeze(dim=1)
