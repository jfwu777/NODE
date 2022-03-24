import torch
import torch.nn as nn
from torch.optim import Adam
import random
import numpy as np
import time
from copy import copy
"""
A very simplified benchmark of Elvet
Toy codes + NO type checks etc.
2 dimensional -> x = x(t), y = y(t)
"""


def init_seed(seed=7):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(10, 1)

    # x -> (x, y)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Model_xy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_x = Model()
        self.model_y = Model()

class BC():
    """
    Boundary Condition
    takes the form that x equals to certain value
    """

    def __init__(self, point, equation, index=0):
        self.point = point
        self.equation = equation
        self.index = index

    def __call__(self, domain, *nth_derivatives):
        return torch.where(self.point == domain, self.equation(domain, *nth_derivatives), torch.zeros_like(domain))

# t is not necessary here
def equation(t, x, y, dx, dy):
    loss1 = sig * (y - x)**2 - dx
    loss2 = -sig * (y - x)**2 - dy
    return loss1, loss2


def solver(
        equations,
        bcs,
        domain,
        order=None,
        model=None,
        optimizer=None,
        epochs=None,
        verbose=False,
        save=False,
        **optimizer_params,
):
    losses = []
    results = {}
    results['x'] = domain
    for epoch in range(epochs):
        domain_x = copy(domain)
        domain_y = copy(domain)
        # y = (x, y)
        x = model.model_x(domain_x)
        y = model.model_y(domain_y)
        if domain_x.grad is not None:
            domain_x.grad.zero_()
        if domain_y.grad is not None:
            domain_y.grad.zero_()

        for i in range(order):
            x.backward(torch.ones_like(x), retain_graph=True)
            dx = torch.clone(domain_x.grad)

            y.backward(torch.ones_like(y), retain_graph=True)
            dy = torch.clone(domain_y.grad)

            # domain.
        loss_f1, loss_f2 = equations(domain, x, y, dx, dy)
        loss_comp2 = bcs(domain, x, y, dx, dy)

        loss_1 = (loss_f1**2 + loss_f2**2).mean() # equation loss, use mean here
        loss_2 = (loss_comp2**2).sum() # BC loss, use sum here
        loss = loss_1 + loss_2
        # KEY POINT: MUST put optimizer.zero_grad() here since y.backward will accumulate gradients on model
        # and that gradient is not for loss, it's for calculating dy therefore shouldn't be put into BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if save:
            torch.save(model.state_dict(), 'my_elvet_ode_toy_sc1.pt')
            # # debug
            # torch.save(results, 'results_sc1.pt')
        if verbose:
            print(epoch, loss)
        else:
            if epoch%1000 == 0:
                print(epoch, loss)
    results['losses'] = losses
    return results


if __name__ == '__main__':
    a = 1
    sig = 1

    init_seed()
    bc = BC(1, lambda t, x, y, dx, dy: (x - 0)**2 + (y - a)**2)
    # domain -> t_eval
    domain = torch.arange(1, 5, 0.0001).reshape(-1, 1)
    domain.requires_grad = True
    model = Model_xy()

    optimizer = Adam(model.parameters(), lr=1e-3)
    results = solver(
        equations=equation,
        bcs=bc,
        domain=domain,
        model=model,
        optimizer=optimizer,
        order=1,
        epochs=10000,
        save=True,
    )
    torch.save(results, 'results-sc1.pt')
