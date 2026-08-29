import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderGRU(nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        Initializes the EncoderGRU module.

        Parameters:
        input_dim (int): the input dimension of the encoder
        emb_dim (int): the embedding dimension of the encoder
        hid_dim (int): the hidden dimension of the encoder
        n_layers (int): the number of layers of the encoder
        dropout (float): the dropout probability of the encoder

        Initializes the embedding, GRU, and dropout layer.
        """
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.gru = nn.GRU(
            input_size = emb_dim,
            hidden_size = hid_dim,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout if n_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, src, src_lengths):

        """
        The forward pass of the EncoderGRU.

        Args:
            src (torch.Tensor): The input tensor.
            src_lengths (torch.Tensor): The lengths of the input sequences.

        Returns:
            sequence_of_output_features (torch.Tensor): The output sequence of the encoder.
            tensor_with_final_hidden_states (torch.Tensor): The final hidden states of the encoder.

        Notes:
            This function takes the input tensor, embeds it, applies dropout, packs the padded sequence, passes it to the GRU, and returns the final hidden states.
        """
        intermediate = self.embedding(src)
        tensor_of_embedded_sequences = self.dropout(intermediate)

        packed_sequence = pack_padded_sequence(
            tensor_of_embedded_sequences,
            src_lengths.cpu(),
            batch_first = True,
            enforce_sorted = False
        )

        packed_sequence_of_output_features, tensor_with_final_hidden_states = self.gru(packed_sequence)

        sequence_of_output_features, _ = pad_packed_sequence(
            packed_sequence_of_output_features,
            batch_first = True
        )

        return sequence_of_output_features, tensor_with_final_hidden_states