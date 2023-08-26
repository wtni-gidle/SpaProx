import torch
import torch.nn as nn
from typing import Optional
from math import sqrt
# 普通的多层感知机，含dropout
class MLP(nn.Module):
    """
    Two-layer feed-forward neural network, using Dropout.
    """
    def __init__(
        self, 
        hidden_sizes: list, 
        num_classes: int = 2, 
        dropout_rate: Optional[list] = None
    ) -> None:
        super().__init__()

        self.num_layers = len(hidden_sizes)

        if dropout_rate is None:
            dropout_rate = [0.2] + [0.5] * (self.num_layers - 2)
        
        assert len(dropout_rate) == self.num_layers - 1, "Length of dropout_rate does not match"

        for i in range(self.num_layers - 1):
            setattr(
                self, 
                "FC_{}".format(i + 1), 
                self.FC_layer(hidden_sizes[i], hidden_sizes[i + 1], dropout_rate[i])
            )

        self.output = nn.Sequential(
            nn.Linear(hidden_sizes[-1], num_classes)
        )

    def FC_layer(self, in_features, out_features, p) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(p = p)
        )

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = getattr(self, "FC_{}".format(i + 1))(x)

        x = self.output(x)

        return x


class Customformer(nn.Module):
    pass


class Attention(nn.Module):
    # 自注意力+一层全连接（含两层dropout），不含layernorm和add
    def __init__(
        self,
        num_heads,
        all_head_size,
        input_size,
        attn_dropout_rate = 0.0,
        proj_dropout_rate = 0.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.all_head_size = all_head_size
        self.head_size = int(all_head_size / num_heads)
        if input_size is None:
            input_size = all_head_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_drop = nn.Dropout(p = attn_dropout_rate)
        self.proj = nn.Linear(self.all_head_size, self.all_head_size)
        self.proj_drop = nn.Dropout(p = proj_dropout_rate)

    def forward(self, input):
        # 获取qkv
        query_layer = self.query(input)
        key_layer = self.key(input)
        value_layer = self.value(input)
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        # 计算注意力分数+dropout
        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / sqrt(self.head_size)
        attn_probs = nn.Softmax(dim = -1)(attn_scores)
        attn_probs = self.attn_drop(attn_probs)
        # 得到注意力输出
        context_layer = torch.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 全连接+dropout
        output = self.proj(context_layer)
        output = self.proj_drop(context_layer)

        return output
    
    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)



class Encoder(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        attn_drop_rate = 0.0, 
        proj_drop_rate = 0.0, 
        batch_first = True, 
        **kwargs
    ) -> None:
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            attn_drop_rate, 
            batch_first = batch_first, 
            **kwargs
        )
        self.dropout = nn.Dropout(p = proj_drop_rate)
    
    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = self.dropout(attn_output)

        return attn_output