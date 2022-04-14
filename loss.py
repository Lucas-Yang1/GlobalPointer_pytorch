import torch
def multilabel_categorical_crossentropy(y_pred: torch.tensor, y_true: torch.tensor):
    """
    B: batch, C: num_classes, k1,k2... 其他的维度
    :param y_pred: B, C, k1, k2....
    :param y_true: shape like y_pred
    :return:
    """

    y_pred = (1-2*y_true)*y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:, :1, ...])
    y_pred_neg = torch.cat([y_pred_neg, zeros],dim=1)
    y_pred_pos = torch.cat([y_pred_pos, zeros],dim=1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=1)

    return neg_loss + pos_loss