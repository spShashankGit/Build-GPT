# # Train the GPT
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 4
context_length = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
n_embed = 32
block_size = 32
# ## 1. Get the dataset


# GPT is a probabilitsic model therefore the response it generate for a prompt is non-deterministic.
with open('dataset/tiny.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# Tiny Shakespeare dataset is used because it is short enought to train on the local computer but long enough i.e. 1 Million characters that it will be hard for a human to cheat/augment the model by providing the answers or regex.


# ### 1. See the dataset


#print('First 500 characters of the dataset:')
#print(text[:500])

# ## 2. Character level Encoding and Decoding Strategy

# Find out how many unique characters are there in this dataset?
# For english text we expect it around 65ish characters (lowercase, uppercase, digits, punctuation, whitespace)
# For code we expect more due to brackets, oprators, special symbold etc.
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Unique characters in the dataset are: ', ''.join(chars))
print('Unique characters in the dataset:', vocab_size)

# Strategy to tokenise the input text
# Assing a integer to the word that is in the dataset. This assignment is based on the vocabulary of possible elements. 
# Because we are building a character level language model, we will asign an integer to each unique character in the dataset. So if A is 1 and then D is 4, word ADD becomes 144


stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encoding = lambda s: [stoi[c] for c in s] # Character -> Integer
decoding = lambda l: ''.join(itos[i] for i in l) # Integer -> Character, this is only a checking step

print(encoding("ADD"))
print(decoding(encoding("ADD")))

## Character level encoding is faily simple to grashp and helps to understand what is under the hood working of the model.
## One popular encoding starategy is SentencePiece which Google uses. Sentece Piece is a sub-word tokenizer which is between encoding each character and encoding entire word.
## Another popular encoding strategy is TikToken which is used by OpenAI. This uses BytePairEncoding tokeniser (BPE)
## In tikToken, instead of 65 tokens it has 50,257 tokens. This encoding strategy was used for GPT 2.
## Why this matter, you can have very long dictionary of words with very small secquence of integers. Or you have a very small dictionary wiht a large sequeence of integers.

# ## 3. Tokenise the dataset based on Encoding strategy defined in previous step

data = torch.tensor(encoding(text), dtype=torch.long)
#print(data.shape, data.dtype)

# %% [markdown]
# ## 3.1 Data Tensor

# %%
#print('CP 1: Data Tensor first 500 characters', data[:500])

# %% [markdown]
# ## 3.2 Train and Test splits

# %%
# It is a good practise to split the dataset into train and test set. This makes sure that the mode is not overfitteing on the dataset and can generalise well. 
# Can the model generalise well and predict for the unknown data is tested by the accuracy of the model. Usually measure by accuracy score.
# There is also sometimes the concept of validation set which is used to tune the hyperparameters of the model in later stages.

n = int(0.9*(len(data)))
train_data = data[:n]
test_data = data[n:]

# In a production environment: The exact split of train and test data is a subject of experiment in iteself. 
# Depending upon the model and the dataset the seplit could be 70:30, 80:20, 90:10 and one must perform experiment will different ration of train and test data to find out the optimal split.
# For this project: 90:10 is taken because the focus is to understand the transformer architecture and the workings of it.

# %% [markdown]
# ## 4. Load the data for training

# %%
# Data loading happends in chunks, this chunks have a max lenghth.
# In a production system: The exact lenght of the chunk is a hyperparamere that means it needs to be tuned by conducting experiements and find out what is the right batch size the leads to least amount of training time vs least amount of loss in training.
# For this project: 256 is taken as the batch size because it is small enough to train on the local computer and large enough to get a good accuracy score.
context_length = 8
train_data[:context_length+1]

# %%
# What we want to do is, train the next character given the previous characters(s)
# example given we have 18, we want to say 47 likely comes next. 
# for 18 & 47 together 56 likely comes next etc.

x = train_data[:context_length]
y = train_data[1:context_length+1]

for t in range(context_length):
    context = x[:t+1]
    target = y[t]
    #print(f'When the input is {context_length} the target is {target}')

# The idea to train the model from 1 to context_length is to make sure the model get use to seeing the different lenght of inputs for infeering the next character.

# %%
# block size is the number of context that will be sent to the model for training.
# Batch is used so that we can make use of the parallel processing power of the GPUs.
# In production system: The bactch size is also a hyperprameter that needs to be tunes by conduting experiements. We are optimising for the efficiency.
torch.manual_seed(1337)

# This function will be used to get the batch for the training and the test set.
def get_batch(split):
    data = train_data if split == 'train' else test_data
    offset = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in offset])
    y = torch.stack([data[i+1:i+context_length+1] for i in offset])
    x,y = x.to(device), y.to(device)
    return x, y

xb,yb = get_batch('train')
# print('Input batch (x):')
# print(xb.shape)
# print(xb)
# print('Target batch (y):')
# print(yb.shape)
# print(yb)

# The learning from out output is that there is a 4x8 array/tensor that is being used to train the model.

# print('' )
for b in range(batch_size):
    for t in range(context_length):
        context = xb[b, :t+1].tolist()
        target = yb[b, t].item()
        #print(f'When the input is {context} the target is {target}')

# %%
#print(xb)

# %% [markdown]
# ## Training hardware is
# MacBook Pro M1 - 2021
# 
# 8-core CPU with 4 performance cores and 4 efficiency cores
# 
# 8-core GPU
# 
# 16-core Neural Engine

# ## 5. Bygram Language Model

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx) # (B,T,C); Batch, Time, Channel
        pos_emb = self.position_embedding_table(torch.arrange(T, device=device)) # (T,c)
        x = tok_emb + pos_emb #(B,T,C)
        logits = self.lm_head(tok_emb) # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            #reshapte the logits
            B,T,C = logits.shape
            logits = logits.view(B*T, C)

            #reshape the target
            targets = targets.view(B*T)
            # quality of a prediction
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    def generate(self,idx, max_new_tokens):
    # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # get the predictions
            logits, loss = self(idx)
            
            # focus only on the last time step
            logits = logits[:,-1, :] # becomes (B,C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    
# m = BigramLanguageModel()
# logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)
# Expectation is -ln(1/65) = 4,17

model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
# Expectation is -ln(1/65) = 4,17

idx = torch.zeros((1,1), dtype=torch.long)
print(decoding(m.generate(idx, max_new_tokens=100)[0].tolist()))

# Create a python optimisser

optimiszer = torch.optim.AdamW(m.parameters(), lr=1e-3)
print(optimiszer)

batch_size = 32
for steps in range(10000):
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)

    # optimize the model
    optimiszer.zero_grad(set_to_none=True)
    loss.backward()
    optimiszer.step()
print(f'train loss {loss.item()}')

# idx = torch.zeros((1,1), dtype=torch.long)
# print(decoding(m.generate(idx, max_new_tokens=100)[0].tolist()))

# idx = torch.zeros((1,1), dtype=torch.long)
# print(decoding(m.generate(idx, max_new_tokens=500)[0].tolist()))
print('----')
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decoding(m.generate(context, max_new_tokens=100)[0].tolist()))