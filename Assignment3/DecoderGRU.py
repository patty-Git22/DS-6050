import torch.nn as nn


class DecoderGRU(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.gru = nn.GRU(
            input_size = emb_dim,
            hidden_size = hid_dim,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout if n_layers > 1 else 0
        )

        self.linear_layer = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden):
        
        input = input.unsqueeze(1)

        intermediate = self.embedding(input)

        tensor_of_embeddings = self.dropout(intermediate)

        tensor_of_output_features, tensor_with_final_hidden_states = self.gru(tensor_of_embeddings, hidden)

        tensor_of_output_features = tensor_of_output_features.squeeze(1)

        tensor_of_logits = self.linear_layer(tensor_of_output_features)

        return tensor_of_logits, tensor_with_final_hidden_states