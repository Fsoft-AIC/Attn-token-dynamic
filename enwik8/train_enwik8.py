
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from definite_reparametrizations import QParametrization, KParametrization, VParametrization

import torch.nn.utils.parametrize as parametrize
import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import time
import os

# constants

BATCH_SIZE = 16
NUM_BATCHES = int(4e5/BATCH_SIZE)
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = float(os.environ['LR'])
VALIDATE_EVERY  = 500
SEQ_LEN = 1024
DIM = 512

# SETTING
ROTARY = False
RESUME = False
step = 0

# helpers
JOB_NAME = os.environ['JOB_NAME']

os.makedirs('out', exist_ok=True)
CKPT_PATH = f'out/{JOB_NAME}.pt'


best_val_loss = float('inf')


def save_checkpoint(model, optimizer, step, loss, path=CKPT_PATH):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'loss': loss
    }, path)
    print(f'Saved new best checkpoint with loss {loss:.4f} at step {step}')

def load_checkpoint(model, optimizer, path=CKPT_PATH):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        loss = checkpoint['loss']
        print(f'Loaded checkpoint from step {step} with loss {loss:.4f}')
        return step, loss
    return 0, float('inf')

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# instantiate GPT-like decoder model

model = TransformerWrapper(
    num_tokens = 256,  # vocab size
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(
        dim = DIM,
        depth = 6,
        heads = 8,
        rotary_pos_emb = ROTARY
    )
)

wneg = float(os.environ["WNEG"])
aneg = float(os.environ["ANEG"])

if "PD" in os.environ and os.environ["PD"] == "1":
    print('registering parametrizations')
    for i in range(0, len(model.attn_layers.layers), 2):
        q_parametrization = QParametrization(DIM)
        parametrize.register_parametrization(
            model.attn_layers.layers[i][1].to_q, 'weight', q_parametrization
        )

        k_parametrization = KParametrization(DIM, wneg, q_parametrization)
        parametrize.register_parametrization(
            model.attn_layers.layers[i][1].to_k, 'weight', k_parametrization
        )

        v_parametrization = VParametrization(DIM, aneg, q_parametrization, k_parametrization)
        parametrize.register_parametrization(
            model.attn_layers.layers[i][1].to_v, 'weight', v_parametrization
        )

model = AutoregressiveWrapper(model)
model.cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    train_x, valid_x = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

if os.path.exists(CKPT_PATH) and RESUME:
    print("Try resuming from last ckpt")
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    loss = checkpoint['loss']
    print(f'Loaded checkpoint from step {step} with loss {loss:.4f}')
    

for i in range(step, NUM_BATCHES):
    begin_time = time.time()
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    run_time = time.time() - begin_time
    print(f'step {i}/{NUM_BATCHES} --- training loss: {loss.item():.4f} --- step time: {run_time:.4f} s --- eta: {(NUM_BATCHES-1-i)*run_time/3600:.4f} h ', end='\r', flush=True)

    if (i+1) % VALIDATE_EVERY == 0:
        model.eval()
        total_val_loss = 0.0
        total_val_batches = 0

        val_loader_full = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False)
        with torch.no_grad():
            for val_batch in val_loader_full:
                val_loss = model(val_batch)
                total_val_loss += val_loss.item()
                total_val_batches += 1

        avg_val_loss = total_val_loss / total_val_batches
        print(f'Average validation loss: {avg_val_loss}')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optim, i, best_val_loss)
