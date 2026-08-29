from Attention import Attention
import torch
import torch.nn as nn


class DecoderGRU(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        Initializes the DecoderGRU module.

        Parameters:
        output_dim (int): the output dimension of the decoder
        emb_dim (int): the embedding dimension of the decoder
        hid_dim (int): the hidden dimension of the decoder
        n_layers (int): the number of layers of the decoder
        dropout (float): the dropout probability of the decoder

        Initializes the embedding, GRU, attention, linear layer, and dropout layer.
        """
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

        self.attention = Attention()
        self.linear_layer = nn.Linear(2 * hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, src_mask):

        """
        Forward pass of the DecoderGRU module.

        Parameters:
            input (torch.Tensor): the input to the decoder with shape (batch_size)
            hidden (torch.Tensor): the hidden state of the decoder with shape (batch_size, hidden_size)
            encoder_outputs (torch.Tensor): the output of the encoder with shape (batch_size, sequence_length, hidden_size)
            src_mask (torch.Tensor, optional): the mask of source tokens with shape (batch_size, sequence_length)

        Returns:
            tuple: (tensor_of_logits, tensor_with_final_hidden_states, matrix_of_attention_weights)
                tensor_of_logits (torch.Tensor): the logits of the decoder with shape (batch_size, output_dim)
                tensor_with_final_hidden_states (torch.Tensor): the final hidden state of the decoder with shape (batch_size, hidden_size)
                matrix_of_attention_weights (torch.Tensor): the attention weights with shape (batch_size, sequence_length)
        """
        input = input.unsqueeze(1)

        intermediate = self.embedding(input)

        tensor_of_embedded_tokens = self.dropout(intermediate)
        
        tensor_of_output_features, tensor_with_final_hidden_states = self.gru(tensor_of_embedded_tokens, hidden)

        tensor_of_output_features = tensor_of_output_features.squeeze(1)

        context_matrix, matrix_of_attention_weights = self.attention(tensor_of_output_features, encoder_outputs, src_mask)

        tensor_of_output_features_and_context = torch.cat((tensor_of_output_features, context_matrix), dim = 1)

        tensor_of_logits = self.linear_layer(tensor_of_output_features_and_context)
    
        return tensor_of_logits, tensor_with_final_hidden_states, matrix_of_attention_weights