import torch
import torch.nn as nn


class EffectiveGlobalPointer(nn.Module):
    def __init__(self, input_size, output_type_size, att_dim, RoPE=True):
        super(EffectiveGlobalPointer, self).__init__()
        self.input_size = input_size
        self.output_type_size = output_type_size
        self.att_dim = att_dim
        self.RoPE = RoPE

        self.W_q = nn.Parameter(torch.ones(self.input_size, self.att_dim))
        self.W_k = nn.Parameter(torch.ones(self.input_size, self.att_dim))

        self.dense = nn.Linear(self.att_dim * 4, self.output_type_size)

        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.xavier_normal_(self.W_q.data)
        nn.init.xavier_normal_(self.W_k.data)

    def get_RoPE(self, q, k):
        """
        :param q: b,l,a
        :param k: b,l,a
        :return:
        """
        cos_pos_em, sin_pos_em = self.sinusoidal_position_embedding(q.size(1), self.att_dim)
        q2 = torch.cat([-q[..., 1::2], q[..., ::2]], dim=-1)
        k2 = torch.cat([-k[..., 1::2], k[..., ::2]], dim=-1)
        q = q * cos_pos_em + q2 * sin_pos_em
        k = k * cos_pos_em + k2 * sin_pos_em

        return q, k

    def get_scores(self, input_features):
        q = torch.matmul(input_features, self.W_q)
        k = torch.matmul(input_features, self.W_k)
        l = q.size(1)
        if self.RoPE:
            q, k = self.get_RoPE(q, k)
        # q, k: [b, l, a]
        s1 = torch.matmul(q, k.transpose(-1, -2))
        # s1: [b, l, l]
        q_k = torch.cat([q, k], dim=-1)
        q_k = torch.cat([q_k.unsqueeze(2).repeat_interleave(l, 2), q_k.unsqueeze(1).repeat_interleave(l, 1)], dim=-1)
        # [b,l,2a] -> [b,l,1,2a], [b,l,2a] -> [b,1,l,2a] 再repeat 再cat
        # q_k : [b, l, l, 4a]
        s2 = self.dense(q_k)
        # s2: [b, l, l, c]
        s2 = s2.permute([0, 3, 1, 2])
        s = s1.unsqueeze(1) + s2

        return s

    def sinusoidal_position_embedding(self, seq_len, output_dim):
        """
        生成旋转位置embedding
        :param seq_len:
        :param output_dim: 需要添加位置embedding的向量的维度
        :return: tensor: [s, dim]
        """
        position_idx = torch.arange(0, seq_len, dtype=torch.float)
        theta_i = torch.pow(10000, -2 * torch.arange(0, output_dim // 2, dtype=torch.float) / output_dim)
        theta_i = theta_i.repeat_interleave(2, 0)
        pos_em = position_idx.unsqueeze(-1) * theta_i.unsqueeze(0)

        cos_pos_em, sin_pos_em = torch.cos(pos_em), torch.sin(pos_em)

        return cos_pos_em.to(self.device), sin_pos_em.to(self.device)

    def forward(self, input_feature, attention_mask, mask_tril=True):
        self.device = input_feature.device

        logits = self.get_scores(input_feature)
        # scores: [b,l,l,c]

        # logits: (batch_size, output_type_size, seq_len, seq_len)

        # attention_mask : batch_size, seq_len

        pad_mask = torch.matmul(attention_mask.unsqueeze(-1).float(),
                                attention_mask.unsqueeze(1).float()).bool().unsqueeze(1)
        # pad_mask: batch_size, 1,  seq_len, seq_len
        if mask_tril:
            triu_mask = torch.ones_like(logits)
            triu_mask = torch.triu(triu_mask).bool()
            # 排除下三角
            logits = logits.masked_fill(~(pad_mask & triu_mask), -1e12)

        else:
            logits = logits.masked_fill(~pad_mask, -1e12)

        # 因为要送入的损失 有logsumexp因此，需要将mask的值调整为负无穷，使得后面的计算exp时，值为0
        # 返回值需不需要归一化方差（ 个人认为是不需要到 ，因为不是计算softmax 即self_attention
        # 不需要考虑其方差大小，并且方差过大应该有利于分类。
        # return logits / self.att_dim ** 0.5
        return logits
