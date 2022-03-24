import torch
import torch.nn as nn
from torch.optim import Adam
import random
import numpy as np
import time
"""
A very simplified benchmark of Elvet
Toy codes + NO type checks etc.
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

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


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

def equation(x, y, dy):
    A = (1 + 3 * x ** 2) / (1 + x + x ** 3)
    return dy + (x + A) * y - x ** 3 - 2 * x - x ** 2 * A


def box(*limits, endpoint=True, dtype=torch.float32):
    if not all([len(limit) == 3 for limit in limits]):
        raise ValueError
    axes = (np.linspace(lower, upper, n_points, endpoint=endpoint) for lower, upper, n_points in limits)
    array = np.vstack([coordinate.flatten() for coordinate in np.meshgrid(*axes)]).T
    return torch.tensor(array, dtype=torch.float32, requires_grad=True)


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

        y = model(domain)
        if domain.grad is not None:
            domain.grad.zero_()
        for i in range(order):
            y.backward(torch.ones_like(domain), retain_graph=True)
            dy = torch.clone(domain.grad)
            # domain.
        loss_comp1 = equations(domain, y, dy)
        loss_comp2 = bcs(domain, y, dy)

        loss_1 = (loss_comp1**2).mean() # equation loss, use mean here
        loss_2 = (loss_comp2**2).sum() # BC loss, use sum here
        loss = loss_1 + loss_2
        # KEY POINT: MUST put optimizer.zero_grad() here since y.backward will accumulate gradients on model
        # and that gradient is not for loss, it's for calculating dy therefore shouldn't be put into BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if save:
            torch.save(model.state_dict(), 'my_elvet_ode_toy.pt')
            # # debug
            # torch.save(results, 'results.pt')
        if verbose:
            print(epoch, loss)
        else:
            if epoch%1000 == 0:
                print(epoch, loss)
    results['losses'] = losses
    return results


if __name__ == '__main__':
    init_seed()
    bc = BC(0, lambda x, y, dy: y - 1)
    domain = box((0, 2, 100))

    model = Model()

    optimizer = Adam(model.parameters(), lr=1e-3)
    results = solver(
        equations=equation,
        bcs=bc,
        domain=domain,
        model=model,
        optimizer=optimizer,
        order=1,
        epochs=50000,
        save=True,
    )
    torch.save(results, 'results.pt')
