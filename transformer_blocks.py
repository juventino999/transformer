import torch
from torch import nn

from typing_extensions import Self, Literal


class Softmax(nn.Module):
    def __init__(self: Self, dim: float, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature == 0:
            raise ValueError('Temperature cannot be zero.')
        self.temperature = temperature
        self.dim = dim

    def forward(self: Self, X: torch.Tensor) -> torch.Tensor:
        # subtract the max for stability with large inputs (thanks, gemini).
        intermediate = torch.e ** (
            (X - torch.max(X, dim=self.dim, keepdim=True)[0]).div(self.temperature)
        )
        # unsqueeze to align dimensions
        # (on dim zero want something like (2, 4, 1) rather than just (2, 4))
        denominator = torch.unsqueeze(intermediate.sum(self.dim), self.dim)
        return intermediate.div(denominator)


class MaskedSoftmax(nn.Module):
    def __init__(self, dim: int, temperature: float = 1.0):
        super().__init__()
        self.softmax = Softmax(dim, temperature)

    def forward(self, X: torch.Tensor, mask: torch.Tensor):
        """Forward pass of masked softmax.

        :param X: Input to be softmaxed rowwise.
        :param mask: Tensor of booleans with same shape as input.
            False masks the value off.
        """
        M = X.masked_fill(~mask, -torch.inf)
        weights = self.softmax(M)
        # if a row is all masked, weights will contain nans
        return torch.nan_to_num(weights, 0)


class Attention(nn.Module):
    def __init__(self, dropout: float = 0.5):
        """Scaled dot product attention with masked softmax.

        :param num_hidden: Number of neurons in the hidden layer.
        :param max_length: Maximum length of a sequence.
        :param dropout: Hyperparameter for dropout.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = MaskedSoftmax(-1)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of attention.

        Number of queries must be equal to the number of keys.

        :param Q: Query tensor of size (n in batch, number of queries, query length).
        :param K: Key tensor (n in batch, number of keys, query length).
        :param V: Value tensor (n in batch, number of keys, value length).
        :param attention_mask: Boolean mask for masked softmax
            (n in batch, number of queries, number of keys).
        """
        sqrt_d = torch.tensor(Q.shape[-1], device=Q.device) ** 0.5
        raw_weights = torch.matmul(Q, torch.transpose(K, -2, -1)).div(sqrt_d)
        attention_weights = self.dropout(self.softmax(raw_weights, attention_mask))
        return torch.matmul(attention_weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_hidden: int,
        num_hidden: int,
        num_heads: int,
        dropout: float = 0.0,
        device: Literal['cpu', 'cuda'] = 'cuda',
    ) -> None:
        """Initializes multi-head attention.

        :param input_hidden: Dimensionality of the input vectors.
        :param num_hidden: Number of neurons in the hidden layer per head.
        :param num_heads: Number of heads to concatenate.
        :param dropout: Attention dropout hyperparameter.
        """
        super().__init__()
        self.input_hidden = input_hidden
        self.num_heads = num_heads
        self.per_hidden = num_hidden
        self.total_hidden = num_hidden * num_heads
        self.Q_w = nn.LazyLinear(self.total_hidden, bias=False, device=device)
        self.K_w = nn.LazyLinear(self.total_hidden, bias=False, device=device)
        self.V_w = nn.LazyLinear(self.total_hidden, bias=False, device=device)
        # restore to input dimensionality
        self.C_w = nn.LazyLinear(self.input_hidden, bias=False, device=device)
        self.attention = Attention(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        query_mask: torch.Tensor,
        key_mask: torch.Tensor,
        causal_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass of MHA.

        Number of queries must be equal to number of keys.

        :param Q: Query tensor of size (n in batch, number of queries,
            query embedding dimension).
        :param K: Key tensor (n in batch, number of keys, key embedding dimension).
        :param V: Value tensor (n in batch, number of keys, value embedding dimension).
        :param query_mask: Boolean mask for masked softmax (n in batch, n in sequence).
        :param key_mask: Boolean mask for masked softmax (n in batch, n in sequence).
        :param causal_attention: Whether to apply an additional causal attention mask.
        """
        if causal_attention is True:
            causal_mask = torch.tril(
                torch.ones(
                    Q.shape[0],
                    Q.shape[1],
                    K.shape[1],
                    device=Q.device,
                    dtype=torch.bool,
                ),
            )
        else:
            causal_mask = torch.ones(
                Q.shape[0], Q.shape[1], K.shape[1], device=Q.device, dtype=torch.bool
            )
        # outer product to create pairwise mask: (batch size, queries, keys)
        matrix_mask = query_mask.unsqueeze(0).transpose(0, 1).transpose(
            1, 2
        ) * key_mask.unsqueeze(0).transpose(0, 1)
        # layer masks, needs to be both causally visible and not padding.
        mask = causal_mask * matrix_mask
        # (batch size*heads, queries, hidden per head)
        q = self.reorder_tensor(self.Q_w(Q))
        k = self.reorder_tensor(self.K_w(K))
        v = self.reorder_tensor(self.V_w(V))
        shaped_mask = torch.repeat_interleave(mask, self.num_heads, 0)
        # still (batch size*heads, queries, hidden per head).
        out = self.attention(q, k, v, shaped_mask)
        # split back into batch size and heads
        z = out.reshape(
            (out.shape[0] // self.num_heads, self.num_heads, -1, self.per_hidden)
        )
        # concat heads -> (batch, queries, hidden per head * heads)
        concatted = z.permute((0, 2, 1, 3)).reshape(z.shape[0], -1, self.total_hidden)
        # apply weights
        return self.C_w(concatted)  # (batch, queries, input dim)

    def reorder_tensor(self, X: torch.Tensor) -> torch.Tensor:
        """Take input of size (batch, n q/k/v, n hidden * n heads) and reshape to
        (batch, n heads, n q/k/v, n hidden).
        """
        intermediate = (
            X
            # (batch size, queries, heads, hidden per head) -> dim 0 grouped by queries
            .reshape(-1, X.shape[1], self.num_heads, self.per_hidden)
            # (batch size, heads, queries, hidden per head) -> dim 0 grouped by heads
            .permute(0, 2, 1, 3)
        )
        return (
            intermediate
            # (batch size*heads, queries, hidden per head) to fit attention
            .reshape(
                (intermediate.shape[0] * intermediate.shape[1], -1, self.per_hidden)
            )
        )

    def __call__(self, *args, **kwds):
        """Forward pass of MHA.

        Number of queries must be equal to the number of keys.

        :param Q: Query tensor of size (n in batch, number of queries,
            query embedding dimension).
        :param K: Key tensor (n in batch, number of keys, key embedding dimension).
        :param V: Value tensor (n in batch, number of keys, value embedding dimension).
        :param query_mask: Boolean mask for masked softmax (n in batch, n in sequence).
        :param key_mask: Boolean mask for masked softmax (n in batch, n in sequence).
        :param causal_attention: Whether to apply an additional causal attention mask.
        """
        return super().__call__(*args, **kwds)


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor):
        """Take an input tensor and add positional encoding data to it.

        :param X: 3D tensor to add 1D (rowwise) positional information to.
        :return: X with positional information added elementwise.
        """

        batch, i, j = X.shape

        cols = torch.repeat_interleave(
            # j step: matrix with column numbers
            torch.arange(0, j, 1, dtype=X.dtype, device=X.device).reshape(1, -1),
            i,
            0,
        )
        # get w for sine columns
        cols[:, ::2] = 1 / (10000 ** (cols[:, ::2] / j))
        # adjust for j - 1 for cosine columns
        cols[:, 1::2] = 1 / (10000 ** ((cols[:, 1::2] - 1) / j))

        rows = torch.repeat_interleave(
            # i step: matrix with row numbers
            torch.arange(0, i, 1, dtype=X.dtype, device=X.device).reshape(-1, 1),
            j,
            1,
        )

        cols = cols * rows

        final = torch.zeros((batch, i, j), device=X.device, dtype=X.dtype)

        final[:, :, ::2] = torch.sin(cols[:, ::2])
        final[:, :, 1::2] = torch.cos(cols[:, 1::2])

        return X + final


class LayerNorm(nn.Module):
    """1D layer normalization with no learnable parameters."""

    def __init__(self):
        super().__init__()

    def forward(self, X: torch.Tensor, dim: int = -1):
        """Perform layer normalization.

        :param X: Tensor to normalize.
        :param dim: Dimension to normalize along.
        :return: Normalized tensor.
        """
        mean = X.mean(dim, keepdim=True)  # equivalent to X.mean(dim).unsqueeze(dim)
        sd = X.std(dim, unbiased=False, keepdim=True)
        return (X - mean) / torch.max(
            sd, torch.tensor(1e-4, device=X.device, dtype=X.dtype)
        )


class ResidualConnection(nn.Module):
    """Residual connection using layer normalization."""

    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm()

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """Forward pass of the residual block.

        :param X: Output to add the residual connection to.
            Layer norm is applied after adding.
        :param Y: Input that is added to the output.
        :return: Tensor with residual connection applied, `layernorm(X) + Y`.
        """
        return self.layernorm(self.dropout(X) + Y)


class FFN(nn.Module):
    """Positionwise feedforward network.

    Applies the same 2 layer MLP to each attention output vector.

    :param num_hidden: Number of output nodes in the hidden layer.
    :param num_output: Number of output nodes in the output layer.
    """

    def __init__(
        self, num_hidden: int, num_output: int, device: Literal['cpu', 'cuda'] = 'cuda'
    ):
        """"""
        super().__init__()
        self.dense_hidden = nn.LazyLinear(num_hidden, device=device)
        self.relu = nn.ReLU()
        self.dense_output = nn.LazyLinear(num_output, device=device)

    def forward(self, X: torch.Tensor):
        """Forward pass of the FFN."""
        z = self.dense_hidden(X)
        z = self.relu(z)
        return self.dense_output(z)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        attention_hidden: int,
        ffn_hidden: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            input_hidden=input_size,
            num_hidden=attention_hidden,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = FFN(num_hidden=ffn_hidden, num_output=input_size)
        self.resid_1 = ResidualConnection(dropout)
        self.resid_2 = ResidualConnection(dropout)

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor):
        z = self.resid_1(self.attention(X, X, X, attention_mask, attention_mask), X)
        return self.resid_2(self.ffn(z), z)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        vocab_size: int,
        input_size: int,
        num_heads: int,
        attention_hidden: int,
        ffn_hidden: int,
        dropout: float = 0,
        device: Literal['cpu', 'cuda'] = 'cuda',
    ):
        super().__init__()
        self.input_size = input_size
        # out size: phrases in batch, tokens in batch, embedding dim for token
        self.embeddings = nn.Embedding(vocab_size, input_size, device=device)
        self.positional_encoding = PositionalEncoding()
        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    input_size, num_heads, attention_hidden, ffn_hidden, dropout
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, X: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass of encoder.

        :param X: Tokenized input sequence.
        :param attention_mask: Attention mask of shape
            (number in batch, number of tokens).
        """
        z = self.positional_encoding(
            self.embeddings(X)
            * torch.sqrt(torch.tensor(self.input_size, device=X.device))
        )
        for block in self.blocks:
            z = block(X=z, attention_mask=attention_mask)
        return z

    def __call__(self, *args, **kwds):
        """Forward pass of encoder.

        :param X: Tokenized input sequence.
        :param attention_mask: Attention mask of shape (number in batch,
            number of tokens).
        """
        return super().__call__(*args, **kwds)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        attention_hidden: int,
        ffn_hidden: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.encoder_decoder_attention = MultiHeadAttention(
            input_hidden=input_size,
            num_hidden=attention_hidden,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.causal_attention = MultiHeadAttention(
            input_hidden=input_size,
            num_hidden=attention_hidden,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = FFN(num_hidden=ffn_hidden, num_output=input_size)
        self.resid_1 = ResidualConnection(dropout)
        self.resid_2 = ResidualConnection(dropout)
        self.resid_3 = ResidualConnection(dropout)

    def forward(
        self,
        X: torch.Tensor,
        decoder_attention_mask: torch.Tensor | Literal[True],
        encoder_out: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        z = self.resid_1(
            self.causal_attention(
                X,
                X,
                X,
                decoder_attention_mask,
                decoder_attention_mask,
                causal_attention=True,
            ),
            X,
        )
        z = self.resid_2(
            self.encoder_decoder_attention(
                z,
                encoder_out,
                encoder_out,
                decoder_attention_mask,
                encoder_attention_mask,
                False,
            ),
            z,
        )
        return self.resid_3(self.ffn(z), z)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        vocab_size: int,
        input_size: int,
        num_heads: int,
        attention_hidden: int,
        ffn_hidden: int,
        dropout: float = 0,
        device: Literal['cpu', 'cuda'] = 'cuda',
    ):
        super().__init__()
        self.input_size = input_size
        # out size: phrases in batch, tokens in batch, embedding dim for token
        self.embeddings = nn.Embedding(vocab_size, input_size, device=device)
        self.positional_encoding = PositionalEncoding()
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    input_size, num_heads, attention_hidden, ffn_hidden, dropout
                )
                for i in range(num_blocks)
            ]
        )
        self.dense = nn.LazyLinear(vocab_size, device=device)

    def forward(
        self,
        X: torch.Tensor,
        decoder_attention_mask: torch.Tensor | Literal[True],
        encoder_out: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ):
        """Forward pass of decoder.

        :param X: Tokenized input sequence (in training) or previously generated
            sequence (in inference).
        :param decoder_attention_mask: Mask for tokens generates thusfar.
        :param encoder_out: Output of encoder.
        :param encoder_attention_mask: Attention mask used to generate encoder output.
        :return: Log odds of each token coming next.
        """
        z = self.positional_encoding(
            self.embeddings(X)
            * torch.sqrt(torch.tensor(self.input_size, device=X.device))
        )
        for block in self.blocks:
            z = block(z, decoder_attention_mask, encoder_out, encoder_attention_mask)
        return self.dense(z)

    def __call__(self, *args, **kwds):
        """Forward pass of decoder.

        :param X: Tokenized input sequence (in training) or previously
            generated sequence (in inference).
        :param decoder_attention_mask: Mask for tokens generates thusfar.
        :param encoder_out: Output of encoder.
        :param encoder_attention_mask: Attention mask used to generate encoder output.
        :return: Log odds of each token coming next.
        """
        return super().__call__(*args, **kwds)


class Transformer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        vocab_size: int,
        input_size: int,
        num_heads: int,
        attention_hidden: int,
        ffn_hidden: int,
        dropout: float = 0,
        device: Literal['cpu', 'cuda'] = 'cuda',
        padding_token: int = 50257,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_blocks,
            vocab_size,
            input_size,
            num_heads,
            attention_hidden,
            ffn_hidden,
            dropout,
            device=device,
        )
        self.decoder = TransformerDecoder(
            num_blocks,
            vocab_size,
            input_size,
            num_heads,
            attention_hidden,
            ffn_hidden,
            dropout,
            device=device,
        )
        self.padding_token = padding_token

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        """Single forward pass.

        :param X: Input sequence (encoder input).
        :param Y: Current output sequence (decoder input).
        :return: Log odds of next token for each token in the decoder output.
        """
        encoder_mask = X != self.padding_token
        encoder_out = self.encoder(X, encoder_mask)
        decoder_mask = Y != self.padding_token
        return self.decoder(Y, decoder_mask, encoder_out, encoder_mask)

    def __call__(self, *args, **kwds):
        """Single forward pass.

        :param X: Input sequence (encoder input).
        :param Y: Current output sequence (decoder input).
        :return: Next token in output sequence.
        """
        return super().__call__(*args, **kwds)
