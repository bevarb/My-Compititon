import torch.nn as nn
import torch

class SEIR_RNN(nn.Module):

    def __init__(self, steps, h, N):
        super().__init__()
        self.N = N
        self.steps = steps
        self.h = h

        self.weights = nn.Parameter(
            torch.tensor([100000, 3, 6, 0, 0.5, 0.2, 0.25], dtype=torch.float32)
            , requires_grad=True)

    def step_do(self, state):
        x = state
        S, E, I, R, A, B, C = (self.weights[0], self.weights[1],
                                    self.weights[2], self.weights[3],
                                    self.weights[4], self.weights[5], self.weights[6])

        _1 = (-1) * B * x[:, 0, 0] * (x[:, 0, 1] + x[:, 0, 2]) / self.N
        _2 = B * x[:, 0, 0] * (x[:, 0, 1] + x[:, 0, 2]) / self.N - A * x[:, 0, 1]
        _3 = A * x[:, 0, 1] - C * x[:, 0, 2]
        _4 = C * x[:, 0, 2]

        _ = torch.stack((_1, _2, _3, _4), dim=-1)

        step_out = x + self.h * torch.clamp(_, -1e5, 1e5)
        return step_out, step_out, A, B, C

    def forward(self, init):
        state = init
        outputs = []
        A, B, C = 0, 0, 0
        for step in range(self.steps):
            step_out, state, A, B, C = self.step_do(state)
            outputs.append(step_out)
        print(A, B, C)
        outputs = torch.stack(outputs, dim=1)

        return outputs