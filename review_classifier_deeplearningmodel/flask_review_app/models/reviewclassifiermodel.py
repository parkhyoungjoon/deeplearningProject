from torch import nn

class reviewClassifierModel(nn.Module):
    def __init__(self, n_vocab, hidden_dim, embedding_dim, n_classes,
                 n_layers, dropout=0.5, bidirectional=True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,         # num_embeddings = vocab이 들어감
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.model = nn.LSTM(
            input_size = embedding_dim,         # Input의 사이즈에 해당하는 수
            hidden_size=hidden_dim,             # 은닉층의 사이즈에 해당하는 수
            num_layers=n_layers,                # RNN의 은닉층 레이어 개수, default = 1
            bidirectional=bidirectional,        # bidrectional True일시 양방향 RNN, default = False
            dropout=dropout,                    # dropout 비율설정 기본값 0
            batch_first=True,                   # True일 경우 Output 사이즈는 (batch, seq, feature) 기본값 False
        )
        if bidirectional:
            self.classifier1 = nn.Linear(hidden_dim*2,n_classes)
            self.classifier2 = nn.Linear(hidden_dim*2,1)
        else:
            self.classifier1 = nn.Linear(hidden_dim,n_classes)
            self.classifier2 = nn.Linear(hidden_dim,1)
        self.dropout = nn.Dropout(dropout)

    def forward(self,inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]  # 마지막 시간 스텝의 출력
        last_output = self.dropout(last_output)

        classesd = self.classifier1(last_output)
        logits = self.classifier2(last_output)

        # LogSoftmax 적용
        classesd = nn.LogSoftmax(dim=1)(classesd)  # 다중 클래스 출력에 LogSoftmax 적용

        return classesd, logits