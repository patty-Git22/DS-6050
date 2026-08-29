import torch


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, src_lengths, tgt = batch
        optimizer.zero_grad()

        tensor_of_logits_for_many_tokens = model(src, src_lengths, tgt)

        number_of_tokens_in_vocabulary = tensor_of_logits_for_many_tokens.shape[-1]
        tensor_of_logits_for_many_tokens = tensor_of_logits_for_many_tokens.reshape(-1, number_of_tokens_in_vocabulary)
        tgt = tgt.reshape(-1)

        loss = criterion(tensor_of_logits_for_many_tokens, tgt)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)