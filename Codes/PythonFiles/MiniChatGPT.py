#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# @title Standaard titeltekst
get_ipython().run_line_magic('pip', 'install tiktoken')
get_ipython().run_line_magic('pip', 'install --upgrade tiktoken')
get_ipython().run_line_magic('pip', 'install --upgrade openai')


# In[ ]:


# Import relevant packages
import torch
import torch.nn as nn
from torch.nn import functional as F
from itertools import islice
import math
import tiktoken
from datetime import datetime


# In[ ]:


#Type of tokenizer BPE = 0, Char = 1, Words = 2
Tokenizer = 0


# In[ ]:


# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
# Wrap the model with DataParallel to use multiple GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 8
n_layer = 8
dropout = 0.2
# -------------

train_val_split = 0.9

torch.manual_seed(3495)


# In[ ]:



# Load data
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open("JoeRoganExperience1139Jordan Peterson.txt", "r") as text_file:
    input_text = text_file.read()


# In[ ]:


if Tokenizer == 0: 

    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-4")
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    vocab_size = num_tokens_from_string(input_text, "cl100k_base")

    
elif Tokenizer == 1:
    
    class simpleTokenizer:
        # Class inherently has two lists, encoded and decoded, used internally
        def __init__(self):
            self.char_to_emb = {}
            self.emb_to_char = {}
    
        # When fit() called on a text, get all characters in that text
        def fit(self, text):
            unique_chars = set(text)
            for i, char in enumerate(sorted(unique_chars)):
                self.char_to_emb[char] = i
                self.emb_to_char[i] = char
    
        def encode(self, text):
            return [self.char_to_emb.get(char, -1) for char in text]
    
        def decode(self, embeddings):
            return ''.join(self.emb_to_char.get(emb, '') for emb in embeddings)
    
    encoding = simpleTokenizer()
    encoding.fit(input_text)
    vocab_size = len(sorted(set(input_text)))       # they cast again to a list

elif Tokenizer == 2:
    
    class simpleTokenizer:
        def __init__(self):
            self.token_to_id = {}  # Generic: works for both char_to_emb and word_to_id
            self.id_to_token = {}  # Generic: works for both emb_to_char and id_to_word
    
        def fit(self, text):
            # For character tokenization, get all unique characters
            #unique_tokens = set(text)  # unique_chars or unique_words based on the input
            words = text.split()
            unique_tokens = set(words)
            for i, token in enumerate(sorted(unique_tokens)):
                self.token_to_id[token] = i
                self.id_to_token[i] = token
    
        def encode(self, text):
            # For character tokenization, iterate over characters
            words = text.split() # for words
            return [self.token_to_id.get(token, -1) for token in words]
    
        def decode(self, ids):
            # Decode a list of ids back to text
            return ' '.join(self.id_to_token.get(id, '') for id in ids)
    
    encoding = simpleTokenizer()
    encoding.fit(input_text)
    vocab_size = len(set(input_text.split()))
    
else:
    print("Invalid Selection")


# In[ ]:



# Define training and test splits
data = torch.tensor(encoding.encode(input_text), dtype = torch.int32)

# 90% train, 10% val
training_data = data[:math.floor(train_val_split*len(data))]
val_data = data[math.floor(train_val_split*len(data)):]

# Generate input-target pairs for a randomly selected batch of data
# Prepares minibatches of sequential data


# In[ ]:



def getRandomBatch(train_or_val):
  if train_or_val == 'train':
    data = training_data
  else:
    data = val_data
  random_indexes = torch.randint(0, (len(data) - block_size), (batch_size,))    # randomly obtain (batch size) 'start' places for the blocks in a batch
  input = torch.stack([data[i:i+block_size] for i in random_indexes])           # stack batches on top of eachother
  target = torch.stack([data[i+1:i+block_size+1] for i in random_indexes])      # target in this instance is defined as the character immediately following the prev.
  input, target = input.to(device), target.to(device).long()    # Needed for CPU/GPU 
  return input, target

getRandomBatch('train')


# In[ ]:


# Evaluates average loss of model for both training and validation data splits

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = getRandomBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# In[ ]:


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)   # Essentially making K' and then sub K' of i to n_head matrices with dim "sequence" x n_embd
        self.query = nn.Linear(n_embd, head_size, bias=False) # Essentially making Q' and then sub Q' of i to n_head matrices with dim "sequence" x n_embd
        self.value = nn.Linear(n_embd, head_size, bias=False) # Essentially making V' and then sub V' of i to n_head matrices with dim "sequence" x n_embd
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # Used later for masking -inf to zero with softmax

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T) # making it a causal system
        wei = self.dropout(wei) # Preventing overfitting I guess
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out # out is


# In[ ]:


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# In[ ]:


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),  # ReLU, but GELU is used in BERT
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), #dropout
        )

    def forward(self, x):
        return self.net(x)


# In[ ]:


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# In[ ]:


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Use vocab_size to correctly initialize embedding tables and other parameters
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, checkpointing=False):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = x.to(device) if not checkpointing else x

        for block in self.blocks:
            if checkpointing:
                # Move x to the GPU for processing this block
                x = x.to(device)
            x = block(x)
            if checkpointing:
                # Move x back to CPU to save GPU memory
                x = x.cpu()

        # Final layer norm and linear layers need to be on GPU
        x = x.to(device)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            # Move targets to GPU for loss calculation
            targets = targets.to(device)
            # Flatten logits and targets for cross-entropy loss
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
     

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


# In[ ]:



# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = getRandomBatch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if abs(losses['train'] - losses['val']) > 1:
        print(f"Stopping early at iteration {iter})")
        break


# In[ ]:


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
#encoded_text = encoding.encode("Left")
#context = torch.tensor([encoded_text], dtype=torch.long, device=device)
gen_text = encoding.decode(m.generate(context, max_new_tokens=2000)[0].tolist())
print(gen_text)


# In[ ]:


# Save the generated text to a file
file_path = 'generated_text.txt'
with open(file_path, 'w') as file:
    file.write(gen_text)

print(f"Generated text saved to: {file_path}")

