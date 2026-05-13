import torch

from pinn.physics.density_matrix import rho_diff
#to się uczy rozwiazania równania różniczkowego

class NNSolver1Q:

    def __init__(self, predictor, solver, data, collocation_points):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predictor = predictor.to(self.device)
        self.solver = solver

        self.data_points = torch.from_numpy(data[0]).to(torch.float32)
        self.data_points = self.data_points.view(-1, 1).to(self.device)

        self.data_values = torch.from_numpy(data[1]).to(torch.complex64)
        self.data_values = self.data_values.squeeze(1).reshape(-1, 2, 2).to(self.device)

        self.collocation_points = torch.from_numpy(collocation_points).to(torch.float32)
        self.collocation_points = self.collocation_points.view(-1, 1)
        self.collocation_points = self.collocation_points.requires_grad_(True)
        self.collocation_points = self.collocation_points.to(self.device)

    def decode_rho(self, out):
        rho = torch.zeros(
            out.shape[0],
            2,
            2,
            dtype=torch.complex64,
            device=out.device
        )

        rho[:, 0, 0] = out[:, 0]
        rho[:, 0, 1] = out[:, 1] + 1j * out[:, 2]
        rho[:, 1, 0] = out[:, 1] - 1j * out[:, 2]
        rho[:, 1, 1] = 1 - out[:, 0]

        return rho

    def physics_residual(self, t):
        out_now = self.predictor(t)
        out_next = self.predictor(t + self.solver.dt)

        rho_now = self.decode_rho(out_now)
        rho_next_nn = self.decode_rho(out_next)

        rho_next_solver = self.solver.step(rho_now)

        return rho_diff(rho_next_nn, rho_next_solver)

    def make_predictions(self, times):
        t = torch.from_numpy(times).to(torch.float32)
        t = t.view(-1, 1).to(self.device)

        out = self.predictor(t)
        rho = self.decode_rho(out)

        return rho.detach().cpu().numpy()