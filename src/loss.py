import torch
import torch.nn.functional as F
import math

# -------------- 熵正则 + 动态 β + top-k 选 token 的完整 loss --------------
def dem_loss(outputs,           # model.forward 的返回对象
                 labels,            # 真实 token id  [B, L]
                 coef=0.2,          # 挑 token 时的熵权重
                 ratio=0.3,         # 选多少比例 token 参与更新
                 beta_base=0.2,     # 熵正则初始强度
                 IGNORE_INDEX=-100,
                 num_items_in_batch=None):
    """
    小众短模板数据专用 loss：
    1. 用 ce - coef*ent 挑最难的 token；
    2. 把负熵（-ent）直接加进 loss，迫使模型保持高 uncertainty；
    3. beta 随 batch 平均熵动态调整，熵越低惩罚越大。
    """
    logits = outputs.logits[:, :-1, :]          # [B, L-1, V]
    labels = labels[:, 1:]                      # [B, L-1]

    # 1. 只保留非 padding 位置
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:                         # 整个 batch 全是 padding
        return torch.tensor(0., device=logits.device, requires_grad=True)

    logits = logits[mask]                       # [N, V]
    labels = labels[mask]                       # [N]

    # 2. 计算概率、交叉熵、熵
    probs = F.softmax(logits, dim=-1)
    log_probs = probs.log()
    ce_tok = F.nll_loss(log_probs, labels, reduction='none')  # [N]
    ent_tok = -(probs * log_probs).sum(-1)                    # [N]

    # 3. 动态 beta：batch 熵越低，beta 越大
    avg_ent = ent_tok.mean().item()
    H_target = 0.4 * math.log(logits.size(-1))          # 经验值
    beta = beta_base * max(1., 1 + (H_target - avg_ent) / H_target)
    beta = min(0.3, beta)                               # 上限保险

    # 4. 挑 token：ce - coef*ent 越大越难
    delta_tok = ce_tok - coef * ent_tok
    N = delta_tok.numel()
    k = max(int(ratio * N), 1)
    _, top_idx = torch.topk(delta_tok, k=k, largest=True, sorted=False)

    # 5. 最终 loss = ce 部分 + 熵正则部分
    loss_ce  = ce_tok[top_idx].mean()
    loss_ent = ent_tok[top_idx].mean() 
    loss = loss_ce + beta * loss_ent

    # 6. 多卡训练时按真实样本数归一化
    if num_items_in_batch is not None:
        denom = max(1, int(num_items_in_batch * ratio))
        loss = loss_ce * k / denom + beta * loss_ent   # 保证 scale 正确

    return loss

# ---------- 内部真正计算损失 ----------
def entropy_loss(outputs, labels, num_items_in_batch=None):
    # outputs.logits: [B, L, V]
    logits = outputs.logits[:, :-1, :]
    labels = labels[:, 1:]

    # 有效 token mask
    IGNORE_INDEX = -100
    mask = labels != IGNORE_INDEX
    logits = logits[mask]           # [N, V]
    labels = labels[mask]           # [N]

    # 交叉熵 & 熵
    probs = F.softmax(logits, dim=-1) # [N, V]
    log_probs = probs.log()           # [N, V]
    ce_tok    = -log_probs.gather(-1, labels[:, None]).squeeze(-1)  # [N]
    ent_tok   = -(probs * log_probs).sum(-1)                   # [N]

    coef   = 0.8
    ratio  = 0.8

    delta_tok = ce_tok - coef * ent_tok                                   # [N]

    # Top-k 选择
    N = delta_tok.numel()
    k = max(int(ratio * N), 1)
    assert k <= N
    _, top_idx = torch.topk(delta_tok, k=k, largest=True, sorted=False)

    if num_items_in_batch is not None:
        loss = ce_tok[top_idx].sum() / int(max(1, num_items_in_batch * ratio))
    else:
        loss = ce_tok[top_idx].mean()

    return loss