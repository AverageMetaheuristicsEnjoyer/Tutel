# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

class LinearTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, **options):
        super().__init__()
        try:
            self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False, dtype=torch.float32 if fp32_gate else None)
        except:
            self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False)
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate

        self.use_penalty = options.pop('use_penalty', False)
        self.srome_alpha = options.pop('srome_alpha', 1.0)
        self.srome_beta = options.pop('srome_beta', 0.9)
        
        self.use_pis = options.pop('use_pis', False)
        self.srome_lambda1 = options.pop('srome_lambda1', 1.0)
        self.srome_lambda2 = options.pop('srome_lambda2', 1.0)

        if self.use_penalty:
            self.register_buffer("avg_logits", torch.zeros(num_global_experts))

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        wg = self.wg.float() if self.fp32_gate else self.wg
        return wg(x.to(dtype=wg.weight.dtype))


Gate = LinearTopKGate
