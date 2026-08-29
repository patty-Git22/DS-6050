import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def forward(self, decoder_hidden, encoder_outputs, src_mask = None):
        """
        Computes the context vector and attention weights based on the decoder hidden state and encoder outputs.

        Args:
            decoder_hidden (torch.Tensor): the hidden state of the decoder with shape (batch_size, hidden_size)
            encoder_outputs (torch.Tensor): the output of the encoder with shape (batch_size, sequence_length, hidden_size)
            src_mask (torch.Tensor, optional): the mask of source tokens with shape (batch_size, sequence_length)

        Returns:
            tuple: (context_matrix, matrix_of_attention_weights)
                context_matrix (torch.Tensor): the context vector with shape (batch_size, hidden_size)
                matrix_of_attention_weights (torch.Tensor): the attention weights with shape (batch_size, sequence_length)
        """
        tensor_of_current_decoder_hidden_states = decoder_hidden.unsqueeze(2)
        
        tensor_of_alignment_scores = torch.bmm(encoder_outputs, tensor_of_current_decoder_hidden_states)

        matrix_of_alignment_scores = tensor_of_alignment_scores.squeeze(2)

        if src_mask is not None:
            matrix_of_alignment_scores = matrix_of_alignment_scores.masked_fill(src_mask == 0, -1e9)

        matrix_of_attention_weights = F.softmax(matrix_of_alignment_scores, dim = 1)

        tensor_of_attention_weights = matrix_of_attention_weights.unsqueeze(1)

        context_tensor = torch.bmm(tensor_of_attention_weights, encoder_outputs)

        context_matrix = context_tensor.squeeze(1)

        return context_matrix, matrix_of_attention_weights
