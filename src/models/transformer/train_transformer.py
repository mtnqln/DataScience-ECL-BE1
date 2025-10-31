import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from src.models.transformer.prepare_data_for_transformer import prepare_data_for_transformer
from src.models.transformer.prepare_data_for_transformer import prepare_transformer

def train_transformer():
    print("Starting training ! \n")
    train_loader,params = prepare_data_for_transformer()
    transformer = prepare_transformer(params=params)
    tgt_vocab_size = params.get("tgt_vocab_size",5000)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" #type: ignore
    print(f"Using device : {device}")

    transformer.train()

    for epoch in tqdm(range(5)):
        for batch in tqdm(train_loader):

            input_ids = batch["src"]
            labels = batch["tgt"]

            optimizer.zero_grad()
            output = transformer(input_ids) # output.shape (batch_size, seq_len, vocab_size)
            """
                we need to flatten the 3D tensor (batch, seq, vocab) → 2D (batch*seq, vocab)
                and the targets (batch, seq) → 1D (batch*seq)

                PyTorch tensors can sometimes be non-contiguous in memory — for example after a .transpose() or a slice.
                That means their underlying memory layout isn't linear, and .view() only works on contiguous tensors.
                .contiguous() ensures the tensor is stored linearly in memory before you reshape it
            """
            loss = loss_func(output.contiguous().view(-1, tgt_vocab_size), labels.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    torch.save(transformer.state_dict(),"src/models/transformer/results/model.pth")
    print("Saved transformer model")