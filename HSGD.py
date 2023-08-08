from torch.optim.optimizer import Optimizer, required
import copy
import torch as t
class HSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Taylor, self).__init__(params, defaults)
        self.history_params = []

    def __setstate__(self, state):
        super(Taylor, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    def step(self, step, t_=0.01, closure=None):
        loss = None
        if closure is not None:
            with t.enable_grad():
                loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            if step < 3:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad
                    param_state = self.state[p]
                    if step == 0:
                        param_state['w_2'] = t.clone(p).detach()
                    if step == 1:
                        param_state['w_1'] = t.clone(p).detach()
                    if step == 2:
                        param_state['w'] = t.clone(p).detach()
                        # param_state['w_p'] = t.clone(d_p).detach()
                    p.data.add_(d_p, alpha=-group['lr'])
            else:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    w = param_state['w']
                    w_1 = param_state['w_1']
                    w_2 = param_state['w_2']
                    # w_p = param_state['w_p']
                    
                    theta_1 = 1.125 * w
                    theta_2 = -0.5 * w_1
                    theta_3 = 0.375 * w_2 
                    theta_4 = 1.25 * group['lr'] * p.grad
                    p.data = theta_1 + theta_2 + theta_3 - theta_4

                    param_state['w_2'] = t.clone( w_1 ).detach()
                    param_state['w_1'] = t.clone( w ).detach()
                    param_state['w'] = t.clone( p ).detach()
                    
        return loss
