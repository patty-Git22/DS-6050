import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderGRU(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(
            input_size =emb_dim,
            hidden_size = hid_dim,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout if n_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, src, src_lengths):
        
        intermediate = self.embedding(src)
        tensoor_of_embedded_sequences = self.dropout(intermediate)

        packed_sequence = pack_padded_sequence(
            tensoor_of_embedded_sequences,
            src_lengths.cpu(),
            batch_first = True,
            enforce_sorted = False
        )

        packed_sequence_of_output_features, tensor_with_final_hidden_states = self.gru(packed_sequence)


        return tensor_with_final_hidden_states