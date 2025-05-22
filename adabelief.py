import math
import torch
from torch.optim import Optimizer

class AdaBelief(Optimizer):
    def __init__(self, params, lr=0.001, beta_1=0.9, beta_2=0.999, lr_dropout=1.0, lr_cos=0, clipnorm=0.0, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("Invalid beta_1 parameter: {}".format(beta_1))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("Invalid beta_2 parameter: {}".format(beta_2))
        if not 0.0 <= lr_dropout <= 1.0:
            raise ValueError("Invalid lr_dropout parameter: {}".format(lr_dropout))
        if not 0.0 <= clipnorm:
            raise ValueError("Invalid clipnorm value: {}".format(clipnorm))
        if not 0.0 <= lr_cos:
            raise ValueError("Invalid lr_cos value: {}".format(lr_cos))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, beta_1=beta_1, beta_2=beta_2, lr_dropout=lr_dropout, lr_cos=lr_cos, clipnorm=clipnorm, eps=eps)
        super(AdaBelief, self).__init__(params, defaults)
        self.iterations = 0

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        self.iterations += 1

        for group in self.param_groups:
            lr = group['lr']
            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            lr_dropout = group['lr_dropout']
            lr_cos = group['lr_cos']
            clipnorm = group['clipnorm']
            eps = group['eps']

            if lr_cos != 0:
                lr *= (math.cos(self.iterations * (2 * math.pi / float(lr_cos))) + 1.0) / 2.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if lr_dropout != 1.0:
                        state['lr_rnd'] = torch.bernoulli(torch.full_like(p.data, lr_dropout))

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Apply gradient clipping if needed
                if clipnorm > 0.0:
                    grad = torch.nn.utils.clip_grad_norm_(p, clipnorm)

                # Update biased first moment estimate
                exp_avg.mul_(beta_1).add_(grad, alpha=1 - beta_1)

                # Update second moment estimate
                grad_diff = grad - exp_avg
                exp_avg_sq.mul_(beta_2).addcmul_(grad_diff, grad_diff, value=1 - beta_2)

                # Compute bias-corrected estimates
                bias_correction1 = 1 - beta_1 ** state['step']
                bias_correction2 = 1 - beta_2 ** state['step']

                adapted_lr = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = (exp_avg_sq.sqrt() + eps)
                step_size = adapted_lr

                step = exp_avg / denom
                if lr_dropout != 1.0:
                    step = step * state['lr_rnd']

                p.data.add_(-step_size, step)

        return loss