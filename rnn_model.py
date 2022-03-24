import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer, batch_first):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
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
        # x = nn.utils.rnn.pack_padded_sequence(x, lengths=length_list, batch_first=opt_batch_first)
        x, _ = self.lstm(x)
        # x, self.hidden = nn.utils.rnn.pad_packed_sequence(x)
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
        x = x[:, -1, :]  # 取 ht 时刻的输出
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(dim=1)
