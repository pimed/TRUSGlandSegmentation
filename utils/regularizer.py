import numpy as np
import torch
from copy import deepcopy

EPS = 1e-8


def get_regularizer(model, model_old, device, old_state, name="ewc"):
    resume = False
    if old_state is not None:
        if name != old_state['name']:
            print(f"Warning: the regularizer you passed {name}"
                  f" is different from the state one {old_state['name']}")
        resume = True

    if name is None:
        return None
    elif name == 'ewc':
        fisher = old_state["fisher"] if resume else None
        return EWC(model, model_old, device, fisher=fisher, alpha=0.9, normalize=False)
    else:
        raise NotImplementedError


def normalize_fn(mat):
    return (mat - mat.min()) / (mat.max() - mat.min() + EPS)


class Regularizer:
    def update(self):
        """ Stub method """
        raise NotImplementedError

    def penalty(self):
        """ Stub method """
        raise NotImplementedError

    def state_dict(self):
        """ Stub method """
        raise NotImplementedError

    def load_state_dict(self, state):
        """ Stub method """
        raise NotImplementedError


class EWC(Regularizer):
    # note: by taking in consideration the torch.distributed package and that the update is only computed by the rank 0,
    #       we can save memory in other ranks. Actually it's not useful because I use GPU with the same memory.
    def __init__(self, model, model_old, device, fisher=None, alpha=0.9, normalize=True):

        self.model = model
        self.device = device
        self.alpha = alpha
        self.normalize = normalize

        # store old model for penalty step
        if model_old is not None:
            self.model_old = model_old
            self.model_old_dict = self.model_old.state_dict()
            self.penalize = True
        else:
            self.penalize = False

        # make the fisher matrix for the estimate of parameter importance
        # store the old fisher matrix (if exist) for penalize step
        if fisher is not None:  # initialize the old Fisher Matrix
            self.fisher_old = fisher
            self.fisher = {}
            for key, par in self.fisher_old.items():
                self.fisher_old[key].requires_grad = False
                self.fisher_old[key] = normalize_fn(par) if normalize else par
                self.fisher_old[key] = self.fisher_old[key].to(device)
                self.fisher[key] = torch.clone(par).to(device)
        else:  # initialize a new Fisher Matrix and don't penalize, we miss an information
            self.fisher_old = None
            self.penalize = False
            self.fisher = {}

        for n, p in self.model.named_parameters():  # update fisher with new keys (due to incremental classes)
            if p.requires_grad and n not in self.fisher:
                self.fisher[n] = torch.ones_like(p, device=device, requires_grad=False)

    def update(self):
        # suppose model have already grad computed, so we can directly update the fisher by getting model.parameters
        for n, p in self.model.named_parameters():
            print(p.grad, self.alpha)
            self.fisher[n] = (self.alpha * (p.grad ** 2)) + ((1 - self.alpha) * self.fisher[n])

    def penalty(self):
        if not self.penalize:
            return 0.
        else:
            loss = 0.
            for n, p in self.model.named_parameters():
                if n in self.model_old_dict and p.requires_grad:
                    loss += (self.fisher_old[n] * (p - self.model_old_dict[n]) ** 2).sum()
            return loss

    def get(self):
        return self.fisher  # return the new Fisher matrix

    def state_dict(self):
        state = {"name": "ewc", "fisher": self.fisher, "alpha": self.alpha,}
        return state

    def load_state_dict(self, state):
        assert state['name'] == 'ewc', f"Error, you are trying to restore {state['name']} into ewc"
        self.fisher = state["fisher"]
        for k,p in self.fisher.items():
            self.fisher[k] = p.to(self.device)
        self.alpha = state["alpha"]