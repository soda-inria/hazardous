"""
modeling_bert.py
"""
import inspect
import math
from typing import Callable, List, Set, Tuple

import torch
from torch import nn

DEFAULT_QUANTILE_HORIZONS = [0.25, 0.5, 0.75]


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings"""

    def __init__(
        self,
        n_numerical_features,
        vocab_size,
        hidden_size=16,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        initializer_range=0.02,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size + 1, hidden_size)
        self.num_embeddings = nn.Parameter(
            torch.randn(1, n_numerical_features, hidden_size)
        )
        self.num_embeddings.data.normal_(mean=0.0, std=initializer_range)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self,
        X_numerical,
        X_categorical,
        X_categ_embeds=None,
    ):
        if X_categ_embeds is None:
            X_categ_embeds = self.word_embeddings(X_categorical)

        X_num_embeds = torch.unsqueeze(X_numerical, 2) * self.num_embeddings
        X_embeds = torch.cat([X_num_embeds, X_categ_embeds], axis=1)
        X_embeds = self.dropout(X_embeds)

        return X_embeds


class BertEncoder(nn.Module):
    def __init__(self, num_hidden_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer() for _ in range(num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        # decide whether or not return attention and hidden states of all layers
        all_hidden_states = [] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_head_mask = head_mask[idx] if head_mask is not None else None

            hidden_states, self_attentions = layer(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_self_attentions.append(self_attentions)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states, all_self_attentions


class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len_dim = 1
        self.chunk_size_feed_forward = 0
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple
        # is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        self_attentions = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        hidden_states = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )

        return hidden_states, self_attentions

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = BertSelfAttention()
        self.output = BertSelfOutput()
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size=16,
        num_attention_heads=2,
        embedding_size=None,
        attention_probs_dropout_prob=0.1,
        position_embedding_type=None,
        max_position_embeddings=512,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0 and embedding_size is None:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of"
                f" attention heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or "absolute"
        if self.position_embedding_type in ["relative_key", "relative_key_query"]:
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * max_position_embeddings - 1, self.attention_head_size
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel
            # forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


class BertSelfOutput(nn.Module):
    def __init__(
        self,
        hidden_size=16,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(
        self,
        hidden_size=16,
        intermediate_size=64,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.functional.gelu  # nn.functional.relu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(
        self,
        hidden_size=16,
        intermediate_size=64,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCLSMulti(nn.Module):
    def __init__(
        self,
        n_features_in,
        n_events,
        n_features_out,
        hidden_size=16,
        intermediate_size=64,
        hidden_dropout_prob=0.0,
    ):
        super().__init__()
        # concatenate embeddings of all features
        net = []

        def w_init(w):
            return nn.init.kaiming_normal_(w, nonlinearity="relu")

        net.append(
            DenseVanillaBlock(
                in_features=hidden_size * n_features_in,
                out_features=intermediate_size,
                batch_norm=True,
                dropout=hidden_dropout_prob,
                activation=nn.ReLU,
                w_init_=w_init,
            )
        )

        # XXX: is this necessary?
        self.net = nn.Sequential(*net)

        net_out = []
        for _ in range(n_events):
            net_out.append(nn.Linear(intermediate_size, n_features_out))
        self.net_out = nn.ModuleList(net_out)
        self.n_events = n_events

    def forward(self, hidden_states, event_of_interest=1):
        hidden_states = hidden_states.flatten(start_dim=1)
        hidden_states = self.net(hidden_states)
        output = self.net_out[event_of_interest - 1](hidden_states)
        return output


class DenseVanillaBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        batch_norm=True,
        dropout=0.0,
        activation=nn.ReLU,
        w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


def prune_linear_layer(
    layer: nn.Linear, index: torch.LongTensor, dim: int = 0
) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which \
        to keep the indices.

    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with \
            :obj:`requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads \
            and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = (
        set(heads) - already_pruned_heads
    )  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the
        # index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors,
) -> torch.Tensor:
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor \
        parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to \
        each chunk independently to save memory.

    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this \
        function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.

    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked

    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`forward_fn`\
            would have given if applied`.


    Examples::

        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states

        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, \
                self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """
    del chunk_size
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"
    tensor_shape = input_tensors[0].shape[chunk_dim]
    assert all(
        input_tensor.shape[chunk_dim] == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method
    #  -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but"
            f" only {len(input_tensors)} input tensors are given"
        )
    return forward_fn(*input_tensors)
