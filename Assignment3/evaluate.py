import torch


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_lengths, tgt = batch

            tensor_of_logits_for_many_tokens = model(src, src_lengths, tgt, teacher_forcing_ratio=0)

            number_of_tokens_in_vocabulary = tensor_of_logits_for_many_tokens.shape[-1]
            tensor_of_logits_for_many_tokens = tensor_of_logits_for_many_tokens.reshape(-1, number_of_tokens_in_vocabulary)
            tgt = tgt.reshape(-1)

            loss = criterion(tensor_of_logits_for_many_tokens, tgt)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)