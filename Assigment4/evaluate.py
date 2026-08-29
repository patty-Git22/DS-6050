import torch


def evaluate(model, iterator, criterion, pad_idx = 0):
    model.eval()
    epoch_loss = 0

    number_of_correct_non_pad_predictions = 0
    number_of_non_pad_target_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_lengths, tgt = batch

            tensor_of_logits_for_many_tokens = model(src, src_lengths, tgt, teacher_forcing_ratio = 0)

            number_of_tokens_in_vocabulary = tensor_of_logits_for_many_tokens.shape[-1]

            matrix_of_predicted_token_indices = tensor_of_logits_for_many_tokens.argmax(dim=-1)

            non_pad_mask = (tgt != pad_idx)

            correct_predictions_mask = (matrix_of_predicted_token_indices == tgt)

            correct_non_pad_predictions_mask = correct_predictions_mask & non_pad_mask

            number_of_correct_non_pad_predictions_for_batch = correct_non_pad_predictions_mask.sum().item()

            number_of_correct_non_pad_predictions += number_of_correct_non_pad_predictions_for_batch
        
            number_of_non_pad_target_tokens += non_pad_mask.sum().item()
            tensor_of_logits_for_many_tokens = tensor_of_logits_for_many_tokens.reshape(-1, number_of_tokens_in_vocabulary)
            tgt = tgt.reshape(-1)

            loss = criterion(tensor_of_logits_for_many_tokens, tgt)

            epoch_loss += loss.item()

    average_loss = epoch_loss / len(iterator)

    if number_of_non_pad_target_tokens > 0:
        token_accuracy = number_of_correct_non_pad_predictions / number_of_non_pad_target_tokens
    else:
        token_accuracy = 0

    return average_loss, token_accuracy