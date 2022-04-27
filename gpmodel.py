import torch
import torch.nn as nn


class GlobalPointer(nn.Module):
    def __init__(self, input_size, output_type_size, att_dim, RoPE=True):
        super(GlobalPointer, self).__init__()
        self.input_size = input_size
        self.output_type_size = output_type_size
        self.att_dim = att_dim
        self.RoPE = RoPE

        # 同时生成 q, k (batch, output_size* att)
        self.dense = nn.Linear(self.input_size, self.output_type_size * self.att_dim * 2)

        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_normal_(self.dense.weight)

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

    def get_scores(self, input_feature):

        batch_size, seq_len = input_feature.size(0), input_feature.size(1)
        input_feature = self.dense(input_feature)
        input_feature = input_feature.view(batch_size, seq_len, self.output_type_size, self.att_dim, 2)
        q, k = input_feature[..., 0], input_feature[..., 1]
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        # q, k: (batch_size, seq_len, output_type_size, self.att) -> (batch_size, output_type_size, seq_len, self.att)

        if self.RoPE:
            q, k = self.get_RoPE(q, k)

        s = torch.matmul(q, k.transpose(-1, -2))
        # s: b,output_size,seq_len, seq_len
        return s

    def sinusoidal_position_embedding(self, seq_len, output_dim):
        """
        生成旋转位置embedding
        :param seq_len:
        :param output_dim: 需要添加位置embedding的向量的维度
        :return:
        """
        position_idx = torch.arange(0, seq_len, dtype=torch.float)
        theta_i = torch.pow(10000, -2 * torch.arange(0, output_dim // 2, dtype=torch.float) / output_dim)
        theta_i = theta_i.repeat_interleave(2, 0)
        pos_em = position_idx.unsqueeze(-1) * theta_i.unsqueeze(0)

        cos_pos_em, sin_pos_em = torch.cos(pos_em), torch.sin(pos_em)
        return cos_pos_em.to(self.device), sin_pos_em.to(self.device)

    def forward(self, input_feature, attention_mask, mask_tril=True):
        """

        :param input_feature: batch, seq_len, features
        :param attention_mask: batch, seq_len
        :return: logits
        """
        self.device = input_feature.device

        logits = self.get_scores(input_feature)

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
