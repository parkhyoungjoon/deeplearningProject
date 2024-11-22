import torch.nn as nn
class AnimeClassifierModel(nn.Module):
    def __init__(self, n_vocab, hidden_dim, embedding_dim, n_classes,
                 n_layers, dropout=0.3, bidirectional=True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(lstm_output_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        # Check the shape of the inputs
        # print(f"Input shape: {inputs.shape}")

        embeddings = self.embedding(inputs)

        # Check the shape after embedding
        # print(f"Embedding shape: {embeddings.shape}")

        lstm_output, _ = self.lstm(embeddings)

        # Check the shape of the LSTM output
        # print(f"LSTM output shape: {lstm_output.shape}")

        last_output = lstm_output[:, -1, :]  # Extract last timestep output
        last_output = self.dropout(last_output)

        # Check the shape after dropout
        # print(f"Last output shape: {last_output.shape}")

        logits = self.classifier(last_output)
        return logits
