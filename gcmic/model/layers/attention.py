import torch
import torch.nn as nn


class DotAttention(nn.Module):
    """ 
    Args:
        dim (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:
            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    Example:
         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dim, attention_type='general', fuse_query=False):
        super(DotAttention, self).__init__()

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dim, dim, bias=False)

        self.fuse_query = fuse_query
        if self.fuse_query:
            self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [bs, query_num, dim]): Sequence of queries to query the context.
            context (:class:`torch.FloatTensor` [bs, context_num, dim]): Data overwhich to apply the attention mechanism.
        Returns:
            :class:`tuple` with `output` and `weights`:
            output (:class:`torch.LongTensor` [bs, query_num, dim]): Tensor containing the attended features.
            weights (:class:`torch.FloatTensor` [bs, query_num, context_num]): Tensor containing attention weights.
        """
        bs, query_num, dim = query.shape
        # context_num = context.shape[1]

        if self.attention_type == "general":
            # query = query.view(bs * query_num, dim)
            query = self.linear_in(query)
            # query = query.view(bs, query_num, dim)

        # (bs, query_num, dim) * (bs, context_num, dim) -> (bs, query_num, context_num)
        scores = torch.bmm(query, context.permute(0, 2, 1))
        # Compute weights across every context sequence
        weights = self.softmax(scores) # (bs, query_num, context_num)
        # weighted output
        out = torch.bmm(weights, context) # (bs, query_num, dim)
 
        if self.fuse_query:
            # concat -> (bs*query_num, 2*dim)
            concat_out = torch.concat([out, query], dim=2)
            concat_out = concat_out.view(bs * query_num, 2 * dim)
            # Apply linear_out on every 2nd dimension of concat
            # output -> (bs, query_num, dim)
            out = self.linear_out(concat_out).view(bs, query_num, dim)
            out = self.tanh(out)

        return out, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, dim=384, num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.query_proj = nn.Linear(in_dim, dim, bias=qkv_bias)        
        self.ctxt_proj = nn.Linear(in_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, context):
        bs, query_num, dim = query.shape
        context_num = context.shape[1]
        q = self.query_proj(query).reshape(bs, query_num, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3) # (bs, num_heads, query_num, head_dim)
        kv = self.ctxt_proj(context).reshape(bs, context_num, 2, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0) # (bs, num_heads, context_num, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (bs, num_heads, query_num, context_num)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(bs, query_num, self.dim) # (bs, query_num, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn


from functools import partial
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(
            self,
            in_dim,
            num_heads=6,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm = norm_layer(in_dim)
        self.drop = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.attn1 = MultiHeadAttention(in_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # self.mlp11 = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.mlp12 = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.attn2 = MultiHeadAttention(in_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # self.mlp21 = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.mlp22 = Mlp(in_features=in_dim, hidden_features=int(in_dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, query, context):
        query = self.norm(query)
        
        context = self.drop(self.attn1(query, self.norm(context))[0])
        # query = query + self.drop(self.mlp11(self.norm(query)))
        context = context + self.drop(self.mlp12(self.norm(context)))
        
        context = context + self.drop(self.attn2(query, self.norm(context))[0])
        # query = query + self.drop(self.mlp11(self.norm(query)))
        context = context + self.drop(self.mlp22(self.norm(context)))
        
        return context, query