"""Implement GPT-2 model from scratch, only using torch."""

import argparse
from collections import OrderedDict
import logging
from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_ort import ORTModule

log = logging.getLogger(Path(__file__).name)

# Model parameters
block_size = 128  # what is the maximum context length for predictions?
eval_interval = 1000
n_blocks = 6
n_heads = 12
n_embd = 252  # the size of the embedding dimension - divides evenly by n_heads
head_embd = n_embd // n_heads
dropout = 0.3


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    """Encode a piece of text into its numerical form: a list of integers."""
    return [stoi[c] for c in text]


def decode(enc: list[int], itos: dict[int, str]) -> str:
    """Decode a piece of text into its numerical form: a list of integers."""
    return "".join(itos[i] for i in enc)


def get_batch(splits: dict[str, torch.Tensor], split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random batch of inputs/targets from the named split."""
    assert split in ("train", "val"), f"split must be either 'train' or 'val', received {split}"
    split_data = splits[split]  # train_data if split == "train" else val_data
    ix = torch.randint(len(split_data) - block_size, size=(batch_size,))  # generate batch_size random starting indices
    x = torch.stack([split_data[i : i + block_size] for i in ix], dim=0)
    y = torch.stack([split_data[i + 1 : i + 1 + block_size] for i in ix], dim=0)
    return x, y


@torch.no_grad()
def estimate_loss(splits: dict[str, torch.Tensor], batch_size: int, eval_iters: int) -> dict[str, torch.Tensor]:
    """Estimate the loss on the train and validation sets."""
    model.eval()
    results = {}
    for split_name in splits:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(splits, split_name, batch_size)
            _, loss = model(x, y)
            losses[k] = loss
        results[split_name] = losses.mean()
    model.train()
    return results


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
        self.heads = nn.ModuleDict(OrderedDict([(str(f"head_{i}"), Head(head_size)) for i in range(n_heads)]))
        self.projection = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through the heads of the multi-head attention mechanism, in parallel -- concatenating their outputs."""
        # Run the heads in parallel, and concatenate their outputs
        # Each head is a self-attention mechanism, so each head is a transformer in its own right.
        # The heads are run in parallel, and their outputs are concatenated together.
        # The output of each head is a tensor of shape (batch, block_size, head_size)
        # The output of the multi-head attention mechanism is a tensor of shape (batch, block_size, n_heads * head_size)
        x = torch.cat([head(x) for name, head in self.heads.items()], dim=-1)
        x = self.projection(x)  # (n_heads * head_size, n_embd)(x(batch, block_size, n_heads * head_size)) -> (B, T, C)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """Implements the feed-forward component of the transformer."""

    def __init__(self, n_embd: int, expansion_factor: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(n_embd, n_embd * expansion_factor)),
                    ("gelu", nn.GELU()),
                    ("linear2", nn.Linear(n_embd * expansion_factor, n_embd)),
                ]
            )
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

        # blocks = [(str(f"block_{i}"), Block(n_heads=n_heads, head_size=head_embd)) for i in range(n_blocks)]
        net_layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(n_blocks):
            name = f"block_{i}"
            net_layers[name] = Block(n_heads=n_heads, head_size=head_embd)
        net_layers["layernorm"] = nn.LayerNorm(n_embd)
        self.net = nn.Sequential(net_layers)
        # self.net = nn.Sequential(OrderedDict([*blocks, ("layernorm", nn.LayerNorm(n_embd))]))
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
        elif torch.all(targets == float("-inf")):  # hack to allow onnxRuntime inference without fallback to pytorch
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
            _hack = torch.ones_like(idx, device=idx.device) * float("-inf")
            logits, loss = self(context)
            # log.warning(">>>>>>>>>>> logits shape: %s -- logits type: %s", logits.shape, type(logits))
            # log.warning(">>>>>>>>>>> loss value:   %s -- loss type:   %s", loss, type(loss))
            # For the bigram model, we only look at the previous character, so take that out and drop a dim
            logits = logits[:, -1, :]  # (B, T, C) becomes (B, C)
            # squash logits into normalised confidences (~probabilities)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Now we can sample from the probability distribution
            new_idx = torch.multinomial(probs, 1)  # (B, 1) --  if pred horizon >1, maybe use replacement=True
            # Append the newly sampled character indices to the context window before predicting in next iteration
            idx = torch.cat([idx, new_idx], dim=1)  # (B, T+1)

        return idx


def load_data() -> tuple[dict[str, torch.Tensor], int, dict[str, int], dict[int, str]]:
    """Create data splits from input file."""

    with open("tinyshakespeare.txt", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    log.info("All chars: %s", "".join(c if c != "\n" else r"\n" for c in chars))
    log.info("vocab_size = %s", vocab_size)

    ## Tokenize - we want to turn the input data into some kind of numeric representation
    # In our case, as a character level language model, we want representations of characters.
    # Alternative libraries: Google's SentencePiece, OpenAI's tiktoken
    stoi = {s: i for i, s in enumerate(chars)}

    itos = dict(enumerate(chars))

    # The full dataset as a tensor - now has shape (N,)
    data = torch.tensor(encode(text, stoi), dtype=torch.long, device=device)
    log.info("data tensor shape = %s", data.shape)

    N = len(text)
    n = int(0.9 * N)
    train_data = data[:n]
    val_data = data[n:]

    log.info("train_data.shape = %s, val_data.shape = %s", train_data.shape, val_data.shape)
    data_splits = {"train": train_data, "val": val_data}

    Xb, Yb = get_batch(data_splits, "train", batch_size=4)
    log.info("Xb.shape = %s, Yb.shape = %s", Xb.shape, Yb.shape)

    return data_splits, vocab_size, stoi, itos


def parse_args() -> argparse.Namespace:
    """Parse command line arguments, otherwise use sensible defaults."""
    parser = argparse.ArgumentParser(description="Train a bigram language model.")
    parser.add_argument("--max_iters", type=int, default=20_000, help="Number of optimisation steps.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Max new tokens to generate.")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Number of eval batches to compute.")
    parser.add_argument("--eval_iters", type=int, default=100, help="Number of eval batches to compute.")
    parser.add_argument("--skip_generate", action="store_true", help="Pass to skip text generation.")
    parser.add_argument("--generate_n_samples", default=3, type=int, help="Number of independent samples to generate.")
    parser.add_argument("--onnx_runtime", action="store_true", help="Use ONNX runtime for training loop.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--cuda", type=int, required=True, help="ID of target cuda device, builds f'cuda:{--cuda}'.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-5s [%(filename)s:%(lineno)-4d] %(message)s",
        level=logging.INFO,
    )

    device = f"cuda:{args.cuda}"
    log.info("Using device: %s", device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    data_splits, vocab_size, stoi, itos = load_data()

    model = BigramLM(vocab_size)
    model = model.to(device)

    if args.onnx_runtime:
        log.info("Wrapping model in ONNX runtime for faster training...")
        model = ORTModule(model)
    else:
        # If we had a newer GPU, we could use pytorch 2.0's torch.compile(model) here.
        log.info("Using native pytorch model...")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info("Model trainable parameters: %s", trainable_params)
    log.info("Model total parameters:     %s", total_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, verbose=True)

    log.info("Training the model...")

    for step in range(args.max_iters + 1):
        xb, yb = get_batch(data_splits, "train", args.batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0:
            log.info("------------------------------------")
            log.info("  step  |  train loss  |  val loss  ")
            log.info("------------------------------------")

        if step % args.eval_interval in (0, args.max_iters):
            eval_results = estimate_loss(data_splits, args.batch_size, args.eval_iters)
            t, v = eval_results["train"].item(), eval_results["val"].item()
            log.info("%s", f"{step:>6}  |   {t:.6f}   |  {v:.6f}")
            scheduler.step(v)

    log.info("Done training!")

    if args.skip_generate:
        log.info("Skipping text generation.")
        sys.exit(0)

    log.info("Generating samples:")
    seed_idx = torch.zeros((args.generate_n_samples, block_size), dtype=torch.long, device=device)
    out = model.generate(idx=seed_idx, max_new_tokens=args.max_new_tokens)

    log.info("------------------------------")
    for generated_batch in range(args.generate_n_samples):
        decoded_lines = decode(out[generated_batch].tolist(), itos).strip().split("\n")
        for line in decoded_lines:
            log.info(line)
        log.info("------------------------------")
