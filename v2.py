
import torch
import torch.nn as nn
import torch.nn.functional as F
from symtable import Class

#hyperramameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
batch_size = 4
context_length = 8

torch.manual_seed(1337)
with open('dataset/tiny.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# Tiny Shakespeare dataset is used because it is short enought to train on the local computer but long enough i.e. 1 Million characters that it will be hard for a human to cheat/augment the model by providing the answers or regex.
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encoding = lambda s: [stoi[c] for c in s] # Character -> Integer
decoding = lambda l: ''.join(itos[i] for i in l) # Integer -> Character, this is only a checking step

print(encoding("ADD"))
print(decoding(encoding("ADD")))

data = torch.tensor(encoding(text), dtype=torch.long)
print(data.shape, data.dtype)

n = int(0.9*(len(data)))
train_data = data[:n]
test_data = data[n:]

context_length = 8
train_data[:context_length+1]

x = train_data[:context_length]
y = train_data[1:context_length+1]

for t in range(context_length):
    context = x[:t+1]
    target = y[t]
    #print(f'When the input is {context} the target is {target}')



# _______________________________
def get_batch(split):
    data = train_data if split == 'train' else test_data
    offset = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in offset])
    y = torch.stack([data[i+1:i+context_length+1] for i in offset])
    x,y = x.to(device), y.to(device)
    return x, y

xb,yb = get_batch('train')

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)      # (B,T,head_size)
        q = self.query(x)    # (B,T,head_size)
        
        # compute attention scores ("affinities")        
        wei = q @ k.transpose(-2 , -1) * C**-0.5  # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T) Do not associate with past
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        # perform the weighted aggregation of the values
        v = self.value(x)     # (B,T,head_size)
        out = wei @ v         # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out
    

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logit for the next token from a lookup table.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd//4) # 4-heads of 8-dimensional self-attention
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)   # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = token_emb + pos_emb  # (B,T,C)
        x = self.sa_head(x) # Apply one head of self-attention (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:] # (B,T)

            # get the predictions for the cropped context only
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,vocab_size)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx
    

m = BigramLanguageModel()
logits, loss = m(xb, yb)
print(logits.shape)
print('Loss is ', loss)

optimiszer = torch.optim.AdamW(m.parameters(), lr=1e-3)
print(optimiszer)

for steps in range(max_iters):
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)

    # optimize the model
    optimiszer.zero_grad(set_to_none=True)
    loss.backward()
    optimiszer.step()
    print(f'train loss {loss.item()}')

print('----')
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decoding(m.generate(context, max_new_tokens=1000)[0].tolist()))