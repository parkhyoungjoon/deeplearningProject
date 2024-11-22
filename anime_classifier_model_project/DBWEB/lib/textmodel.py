import torch.nn as nn

class TextModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_list, act_func, model, num_layers=1):
        super().__init__()
        # 입력층 (LSTM)
        if model == 'lstm':
            self.lstm_layer = nn.LSTM(input_size, hidden_list[0], num_layers, batch_first=True)
        elif model == 'rnn':
            self.rnn_layer = nn.RNN(input_size, hidden_list[0], num_layers, batch_first=True)
        elif model == 'gru':
            self.gru_layer = nn.GRU(input_size, hidden_list[0], num_layers, batch_first=True)
        # 은닉층
        self.hidden_layer_list = nn.ModuleList()
        for i in range(len(hidden_list)-1):
            self.hidden_layer_list.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
        # 출력층
        self.output_layer = nn.Linear(hidden_list[-1], output_size)

        self.act_func = act_func
        self.dropout = nn.Dropout(0.5)
        self.model = model
        
    def forward(self, x):
        # 입력층
        if self.model == 'lstm':
            lstm_out, (hn, cn) = self.lstm_layer(x) # lstm_out : 모든 타입스텝 출력
            x = lstm_out[:, -1, :] # 마지막 타입스텝 출력
        elif self.model == 'rnn':
            rnn_out, hn = self.rnn_layer(x) # rnn_out : 모든 타입스텝 출력
            x = rnn_out[:, -1, :] # 마지막 타입스텝 출력
        elif self.model == 'gru':
            gru_out, hn = self.gru_layer(x) # gru_out : 모든 타입스텝 출력
            x = gru_out[:, -1, :] # 마지막 타입스텝 출력
        # 은닉층
        for layer in self.hidden_layer_list:
            x = layer(x)
            x = self.act_func(x)
            x = self.dropout(x)
        # 출력층
        return self.output_layer(x) # 로짓값