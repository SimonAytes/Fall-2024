import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from transformers import AdamW, get_scheduler
from base import eli5_dataset, set_seed
import numpy as np

class SimplifiedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout=0.1, max_len=200):
        super(SimplifiedTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.register_buffer("positional_encoding", self.create_positional_encoding(d_model, max_len))
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def create_positional_encoding(d_model, max_len):
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids):
        # Ensure positional encoding is on the correct device
        positional_encoding = self.positional_encoding.to(input_ids.device)
        embeddings = self.embedding(input_ids) + positional_encoding[:, :input_ids.size(1), :]
        tgt = embeddings.permute(1, 0, 2)  # [seq_length, batch_size, d_model]
        memory = torch.zeros_like(tgt, device=input_ids.device)  # Decoder-only setup
        transformer_output = self.transformer_decoder(tgt=tgt, memory=memory)
        logits = self.lm_head(transformer_output.permute(1, 0, 2))  # [batch_size, seq_length, vocab_size]
        return logits

def train(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        inputs = batch.to(device)
        targets = inputs.clone()

        optimizer.zero_grad()
        outputs = model(inputs, labels=targets)  # Fine-tuning GPT-2 includes labels directly
        loss = outputs.loss / gradient_accumulation_steps  # Loss is computed internally by GPT-2
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            outputs = model(inputs)
            logits_list.append(outputs.logits.cpu().numpy())
    return np.concatenate(logits_list)

# Set random seed
set_seed(seed=0)

# Hyperparameters
BATCH_SIZE = 8  # Smaller batch size for better gradient updates
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5  # Fine-tuning-specific LR
WEIGHT_DECAY = 0.01
EPOCHS = 5
WARMUP_STEPS = 100

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and datasets
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
trainset = eli5_dataset(tokenizer, 200, "train")
testset = eli5_dataset(tokenizer, 200, "test")

train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

# Initialize pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Adjust for the tokenizer size if needed
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = get_scheduler(
    "cosine", optimizer=optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=EPOCHS * len(train_loader)
)

# Training loop
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, scheduler, device, GRADIENT_ACCUMULATION_STEPS)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss}")

# Evaluate on test set
logits = evaluate(model, test_loader, device)

# Save logits
np.save("20233960.npy", logits)


# LOSS LOGGING:
# Epoch 1/5, Loss: 3.671701571279162
# Epoch 2/5, Loss: 3.532481182739645
# Epoch 3/5, Loss: 3.4548448300977395
# Epoch 4/5, Loss: 3.396752516716705
# Epoch 5/5, Loss: 3.351113319613029
