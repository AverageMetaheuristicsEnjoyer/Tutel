# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch.distributions.normal import Normal

def _one_hot_with_dtype(data, num_classes, dtype, hot_value=1):
    result = torch.zeros([data.size(0), num_classes], device=data.device, dtype=dtype)
    result.scatter_(1, data.unsqueeze(-1), hot_value)
    return result

def gshard_loss(
    scores_w_noise,
    top_ids,
    use_pis=False,
    router_probs=None,
    top_k_scores=None,
    num_groups=0,
    experts_per_group=0,
    lambda1=1.0,
    lambda2=1.0
    ):
    num_samples, num_global_experts = int(scores_w_noise.size(0)), int(scores_w_noise.size(1))
    mask = _one_hot_with_dtype(top_ids[:, 0], num_global_experts, dtype=scores_w_noise.dtype,
        hot_value=num_global_experts / num_samples)
    me = torch.sum(scores_w_noise, dim=0)
    ce = torch.sum(mask, dim=0)
    l_aux = torch.sum(me * ce) / num_samples
    if use_pis:
        assert router_probs is not None
        assert top_k_scores is not None
        assert num_groups > 0
        
        batch_size = router_probs.shape[0]
        top_k_per_group = top_k_scores.shape[1] // num_groups

        grouped_router_probs = router_probs.view(batch_size, num_groups, experts_per_group)
        grouped_top_k_scores = top_k_scores.view(batch_size, num_groups, top_k_per_group)

        norm_pi_per_group = torch.linalg.vector_norm(grouped_router_probs, ord=2, dim=-1).pow(2)
        norm_pi_tilde_per_group = torch.linalg.vector_norm(grouped_top_k_scores, ord=2, dim=-1).pow(2)

        pis_loss = -lambda1 * torch.mean(norm_pi_per_group) + lambda2 * torch.mean(norm_pi_tilde_per_group)
        
        l_aux += pis_loss
    return l_aux

def load_importance_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
    def load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
        assert gate_noise > 0, "`gate_noise` must be > 0 for normalization in load_importance_loss()."
        normal = Normal(
            torch.tensor([0.0], device=scores_wo_noise.device),
            torch.tensor([gate_noise / num_global_experts], device=scores_wo_noise.device),
        )
        threshold = topk_logits[:, -1].view(-1, 1).float()
        diff = scores_wo_noise.float() - threshold.float()
        prob = normal.cdf(diff)
        Load = prob.sum(0)
        l_load = Load.float().var() / (Load.float().mean() ** 2 + 1e-10)
        return l_load

    def importance_loss(scores_wo_noise):
        Impi = scores_wo_noise.float().sum(0)
        l_imp = Impi.float().var() / (Impi.float().mean() ** 2 + 1e-10)

        return l_imp

    l_imp = importance_loss(scores_wo_noise)
    l_load = load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise)
    return (l_imp + l_load) / 2.0