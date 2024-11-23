import torch.nn as nn
import torch
from panel.io.embed import save_dict


#自注意力板块
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):       #类初始化，embed-size为嵌入的总维度，heads为头的数量
        super(SelfAttention, self).__init__()    #调用继承父类初始化方法
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads      # 每个注意力头的维度

        assert (self.head_dim * heads == embed_size)    # 确保总的嵌入维度能被头数整除

        # 线性层
        # 定义用于计算值（values）、键（keys）、查询（queries）的线性变换
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 定义用于将多头注意力输出拼接并映射回嵌入维度的线性变换
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        """
                前向传播，计算多头自注意力。
                :param values: 值向量 V，形状为 (N, value_len, embed_size)
                :param keys: 键向量 K，形状为 (N, key_len, embed_size)
                :param query: 查询向量 Q，形状为 (N, query_len, embed_size)
                :param mask: 掩码，形状为 (N, 1, 1, key_len)，可选
                :return: 输出，形状为 (N, query_len, embed_size)
                """
        N = query.shape[0] # 获取批量的大小（样本数）
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 将嵌入维度拆分为多个头，每个头的形状为 (N, seq_len, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 对值、键和查询应用线性变换以生成新的 V、K、Q
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # 矩阵乘法，使用爱因斯坦标记法
        # 使用爱因斯坦求和公式计算点积注意力得分

        if mask is not None:# 如果提供了掩码，则将掩码位置的注意力得分置为极小值（如 -1e20），防止注意力集中于这些位置
            energy = energy.masked_fill(mask==0, float("-1e20"))

        # 对注意力得分进行缩放并计算 softmax，以获得注意力权重
        # 缩放因子为 sqrt(embed_size)，防止点积结果过大导致梯度不稳定
        # 注意力权重的形状为 (N, heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        # 使用注意力权重对值向量进行加权求和
        # attention 的形状为 (N, heads, query_len, key_len)
        # values 的形状为 (N, value_len, heads, head_dim)
        # 结果 out 的形状为 (N, query_len, heads, head_dim)
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values])

        # 将最后两个维度（heads 和 head_dim）拼接，恢复为 (N, query_len, embed_size)
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # 通过全连接层将输出映射回嵌入维度，形状为 (N, query_len, embed_size)
        out = self.fc_out(out)
        return out


# transformer块
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        """
               初始化 TransformerBlock。
               :param embed_size: 输入嵌入的维度。
               :param heads: 多头注意力的头数。
               :param dropout: Dropout 的概率，用于正则化。
               :param forward_expansion: 前向传播网络的扩展因子，控制隐藏层的大小。
               """
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)# 初始化多头自注意力模块

        self.norm1 = nn.LayerNorm(embed_size)# 初始化两个层归一化模块，用于规范化注意力输出和前向传播输出
        self.norm2 = nn.LayerNorm(embed_size)

        # 定义前向传播模块（Feed-Forward Network）
        # 包括两层线性层，中间插入 ReLU 激活函数
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size), # 扩展特征维度
            nn.ReLU(), # 引入非线性
            nn.Linear(forward_expansion*embed_size, embed_size) # 缩回原始维度
        )
        # Dropout，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        """
               前向传播过程。
               :param value: 值向量 (N, value_len, embed_size)。
               :param key: 键向量 (N, key_len, embed_size)。
               :param query: 查询向量 (N, query_len, embed_size)。
               :param mask: 掩码，用于屏蔽某些位置，形状为 (N, 1, 1, key_len)。
               :return: 输出张量，形状为 (N, query_len, embed_size)。
               """
        # 1. 计算多头自注意力
        attention = self.attention(value, key, query, mask)
        # 使用残差连接和层归一化
        x = self.dropout(self.norm1(attention + query))

        # 2. 前向传播网络
        forward = self.feed_forward(x)
        # 再次使用残差连接和层归一化
        out = self.dropout(self.norm2(forward + x))

        return out

# 编码器
class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,     # 源语言词汇表大小
                 embed_size,         # 嵌入维度
                 num_layers,         # Transformer Block 的层数
                 heads,              # 多头注意力的头数
                 device,             # 运行设备（CPU 或 GPU）
                 forward_expansion,  # 前向传播网络扩展倍数
                 dropout,            # Dropout 概率
                 max_length          # 最大序列长度
                 ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size # 嵌入维度
        self.device = device         # 设备

        # 词嵌入层，将单词索引映射为嵌入向量
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        # 位置嵌入层，用于提供序列中每个位置的位置信息
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # 定义多个 Transformer Block，并存入一个 ModuleList 容器
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
            for _ in range(num_layers)] # 创建 num_layers 个 Transformer Block
        )

        # Dropout 层，用于正则化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
                前向传播逻辑。
                :param x: 输入序列，形状为 (N, seq_length)，其中 N 为批量大小。
                :param mask: 掩码，用于屏蔽某些位置，形状为 (N, 1, 1, seq_length)。
                :return: 编码后的输出，形状为 (N, seq_length, embed_size)。
                """

        # 获取输入序列的批量大小和序列长度
        N, seq_lengh = x.shape

        # 生成位置索引张量，形状为 (N, seq_length)
        positions = torch.arange(0, seq_lengh).expand(N, seq_lengh).to(self.device)

        # 计算词嵌入与位置嵌入的和，并应用 Dropout
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # 依次通过每一层 Transformer Block
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# 解码器块
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


# 解码器
class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


# 完整的transformer
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 heads=8,
                 dropout=0,
                 device="cuda",
                 max_length=100
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(x, trg[:, :-1])
    print(out)
    print(out.shape)
    save_dict = './transformer.pth'
    torch.save(model,save_dict)

