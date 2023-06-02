"""Implement GPT-2 model from scratch, only using torch."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(Path(__file__).name)
logging.basicConfig(
    format="%(asctime)s %(levelname)-5s [%(filename)s:%(lineno)-4d] %(message)s",
    level=logging.INFO,
)

# Define hyperparameters at module level
batch_size = 8  # how many independent sequences will we process in parallel?
block_size = 32  # what is the maximum context length for predictions?
max_iters = 12000
eval_interval = 500
learning_rate = 1e-3
eval_iters = 100
n_blocks = 3
n_heads = 6
n_embd = 240  # the size of the embedding dimension - divides evenly by n_heads
head_embd = n_embd // n_heads
n_layer = 6
dropout = 0.2

device = "cpu"  # "mps" if (torch.has_mps and torch.backends.mps.is_available()) else "cpu"
log.info("Using device: %s", device)

torch.manual_seed(1337)

with open("tinyshakespeare.txt", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
log.info("All chars: %s", "".join(c if c != "'\n'" else r"\n" for c in chars))
log.info("vocab_size = %s", vocab_size)


## Tokenize - we want to turn the input data into some kind of numeric representation
# In our case, as a character level language model, we want representations of characters.
# Alternative libraries: Google's SentencePiece, OpenAI's tiktoken
stoi = {s: i for i, s in enumerate(chars)}

itos = dict(enumerate(chars))


def encode(text: str) -> list[int]:
    """Encode a piece of text into its numerical form: a list of integers."""
    return [stoi[c] for c in text]


def decode(enc: list[int]) -> str:
    """Decode a piece of text into its numerical form: a list of integers."""
    return "".join(itos[i] for i in enc)


# The full dataset as a tensor - now has shape (N,)
data = torch.tensor(encode(text), dtype=torch.long, device=device)
log.info("data tensor shape = %s", data.shape)

N = len(text)
n = int(0.9 * N)
train_data = data[:n]
val_data = data[n:]

log.info("train_data.shape = %s, val_data.shape = %s", train_data.shape, val_data.shape)


def get_batch(split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random batch of inputs/targets from the named split."""
    assert split in ("train", "val"), f"split must be either 'train' or 'val', received {split}"
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,))  # generate batch_size random starting indices
    x = torch.stack([data[i : i + block_size] for i in ix], dim=0)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix], dim=0)
    return x, y


@torch.no_grad()
def estimate_loss() -> dict[str, torch.Tensor]:
    """Estimate the loss on the train and validation sets."""
    model.eval()
    splits = ["train", "val"]
    results = {}
    for split_name in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_name, batch_size)
            _, loss = model(x, y)
            losses[k] = loss
        results[split_name] = losses.mean()
    model.train()
    return results


Xb, Yb = get_batch("train", batch_size=4)

log.info("Xb.shape = %s, Yb.shape = %s", Xb.shape, Yb.shape)


class Head(nn.Module):
    """Single self-attention head."""

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # This is a tensor that is used to mask out the upper triangular part of the attention matrix.
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.mask: torch.Tensor  # allows mypy to use correct type
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the self-attention head.

        Args:
          x: input sequence of shape (batch, block_size, n_embd)

        Returns:
          the output sequence of shape (batch, block_size, n_embd)
        """
        B, T, C = x.shape
        # Compute the keys, queries and values. Each token's (1, C) vector remains private to itself here -
        # information is not shared. between the individual samples of the input batch.
        # This is self attention, so tokens of the input are each "projected" into three semantic parts:
        # 1. Queries: the information I find interesting want to pay attention to.
        # 2. Keys:    the information I have to offer
        # 3. Values:  the information I will give you if you find me interesting i.e. we share high affinity.
        q: torch.Tensor = self.query(x)  # Linear(n_embd, head_size)(x(batch, block_size, n_embd)) -> (B, T, C)
        k: torch.Tensor = self.key(x)  # Linear(n_embd, head_size)(x(batch, block_size, n_embd)) -> (B, T, C)
        v: torch.Tensor = self.value(x)  # Linear(n_embd, head_size)(x(batch, block_size, n_embd)) -> (B, T, C)
        # Applying a linear layer to 3d tensor applies its weights to the last dimension.

        # Compute the weighted average of the values, using the scaled dot-product attention
        # Note that we mask out the upper triangular part of the attention matrix, so that
        # each position can only attend to previous positions.
        # This is done by adding a large negative number to the upper triangular part of the attention matrix.
        # The softmax of a large negative number is close to zero, so the corresponding weights are close to zero.
        # This is equivalent to not attending to those positions. Normalise by the embedding size to
        attention_scores = q @ k.transpose(-2, -1) * self.head_size**-0.5  # shapes: (B, T, h)@(B, h, T) --> (B, T, T)
        # Having T, T here is why the attention mechanism is quadratic in the input sequence length!

        # Apply the mask to fill the upper triangular part of the attention matrix with a large negative number.
        # attention_scores = attention_scores.masked_fill(self.mask[:T, :T] == 0, -1e9)  # noqa: ERA001
        attention_scores.masked_fill_(self.mask[:T, :T] == 0, float("-inf"))  # inplace equivalent

        # Normalise the attention scores across the last dimension (the T dimension) using the softmax function.
        affinities = F.softmax(attention_scores, dim=-1)  # (B, T, T)
        affinities = self.dropout(affinities)  # (B, T, T)
        return affinities @ v  # (B, T, T)@(B, T, h) -> (B, T, h)


class MultiHeadAttention(nn.Module):
    """Implementation of the multi-head self-attention mechanism."""

    # initialise the multi-head attention mechanism
    def __init__(self, n_heads: int, head_size: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_size = head_size
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through the heads of the multi-head attention mechanism, in parallel -- concatenating their outputs."""
        # Run the heads in parallel, and concatenate their outputs
        # Each head is a self-attention mechanism, so each head is a transformer in its own right.
        # The heads are run in parallel, and their outputs are concatenated together.
        # The output of each head is a tensor of shape (batch, block_size, head_size)
        # The output of the multi-head attention mechanism is a tensor of shape (batch, block_size, n_heads * head_size)
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.projection(x)  # (n_heads * head_size, n_embd)(x(batch, block_size, n_heads * head_size)) -> (B, T, C)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """Implements the feed-forward component of the transformer."""

    def __init__(self, n_embd: int, expansion_factor: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * expansion_factor),
            nn.GELU(),
            nn.Linear(n_embd * expansion_factor, n_embd),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feed-forward component of the transformer.

        Args:
          x: input sequence of shape (batch, block_size, n_embd).
        """
        x = self.net(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Implements a transformer block."""

    def __init__(self, n_heads: int, head_size: int) -> None:
        super().__init__()
        # Apply pre-norm formulation of layer normalisation,
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.multi_head_self_attention = MultiHeadAttention(n_heads, head_size)
        self.feed_forward = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transformer block.

        Args:
          x: input sequence of shape (batch, block_size, n_embd).

        Returns:
          the output sequence of shape (batch, block_size, n_embd).
        """
        # Skip connections are added around the attention and feed-forward components.
        x = x + self.multi_head_self_attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Implementation of the Gaussian Error Linear Unit (GELU) activation function.

    GELU is a smooth approximation to relu, with a small area allowing non-zero gradient for negative inputs.

    See: https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x * 2.0**-0.5))


class BigramLM(nn.Module):
    """Implements the bigram language model as a LUT of learned vectors for each bigram."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # Define a LUT for the bigram embeddings
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd)
        blocks = [Block(n_heads=n_heads, head_size=head_embd) for _ in range(n_blocks)]
        self.net = nn.Sequential(
            *blocks,
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_heads * head_embd, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of the bigram language model.

        1. index to get embeddings of each input character (given as their indices idx)
        2. treat those embeddings as the logits across the vocab, thus predicted likelihood of the next char

        Args:
          idx: indices of the input character sequences, (B, block_size).
          targets: indices of the characters that are to be predicted, (B, block_size). [Optional]

        Returns:
          the predicted logits, and the loss value if targets were given.  shape (batch*block, embedding_size)
        """
        B, T = idx.shape  # batch size, block size
        token_embed = self.token_embedding_table(idx)  # returns (B, block_size, vocab_size) a.k.a. (B, T, C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_embed + pos_embed  # returns (B, block_size, embedding_size) a.k.a. (B, T, C)
        x = self.net(x)  # returns (B, block_size, embedding_size) a.k.a. (B, T, C)
        logits = self.lm_head(x)  # returns (B, block_size, vocab_size) a.k.a. (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # batch size, block size, vocab size. block_size is the context length/window
            logits = logits.view(B * T, C)
            # torch's cross_entropy requires shape (N, C), all logits for a sample as final dim
            # compute the loss - use cross entropy loss (negative log likelihood)
            targets = targets.view(B * T)  # make the shape of targets match logits for F.cross_entropy to work
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Given an existing context, generate the next tokens based on likeliness of the bigram model."""
        for _ in range(max_new_tokens):
            # Ensure that we never pass in more than the context window size
            context = idx[:, -block_size:]  # (B, T)
            # compute logits from the model - loss is None as we don't supply targets
            logits, loss = self(context)
            # For the bigram model, we only look at the previous character, so take that out and drop a dim
            logits = logits[:, -1, :]  # (B, T, C) becomes (B, C)
            # squash logits into normalised confidences (~probabilities)
            probs = F.softmax(logits, dim=1)  # (B, C)
            # Now we can sample from the probability distribution
            new_idx = torch.multinomial(probs, 1)  # (B, 1) --  if pred horizon >1, maybe us replacement=True
            # Append the newly sampled character indices to the context window before predicting in next iteration
            idx = torch.cat([idx, new_idx], dim=1)  # (B, T+1)

        return idx


# compare activation functions.
# xs = torch.linspace(-5, 5, 5000)
# fig, axs = plt.subplots(1, 1, figsize=(8, 6))
# axs.plot(xs, torch.tanh(xs), lw=5, alpha=0.5)
# axs.plot(xs, torch.nn.functional.relu(xs), lw=5, alpha=0.5)
# axs.plot(xs, gelu(xs), lw=5, alpha=0.9)
# axs.plot(xs, torch.sigmoid(xs), lw=5, alpha=0.5)
# axs.legend(["tanh", "relu", "gelu", "gelu_exact"])
# # set xlimits to be -2, +2:
# axs.set_xlim(-4, 4)
# axs.set_ylim(-2, 4)
# axs.set_title("Activation functions")
# plt.show()


if __name__ == "__main__":
    model = BigramLM(vocab_size)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info("Model trainable parameters: %s", trainable_params)
    log.info("Model total parameters:     %s", total_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    log.info("Training the model...")

    log.info("------------------------------------")
    log.info("  step  |  train loss  |  val loss  ")
    log.info("------------------------------------")

    for step in range(max_iters + 1):
        xb, yb = get_batch("train", batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval in (0, max_iters):
            eval_results = estimate_loss()
            t, v = eval_results["train"].item(), eval_results["val"].item()
            log.info("%s", f"{step:>6}  |   {t:.6f}   |  {v:.6f}")

    log.info("Done training!")

    log.info("Generating samples:")
    num_samples = 3
    seed_idx = torch.zeros((num_samples, block_size), dtype=torch.long, device=device)
    out = model.generate(idx=seed_idx, max_new_tokens=500)

    log.info("------------------------------")
    for generated_batch in range(num_samples):
        decoded_text = decode(out[generated_batch].tolist()).strip()
        for line in decoded_text.split("\n"):
            log.info(line)
        log.info("------------------------------")
