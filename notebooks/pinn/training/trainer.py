import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Trainer:

    def __init__(
        self,
        nn_solver,
        epochs,
        learning_rate,
        physics_loss=True
    ):

        self.nn_solver = nn_solver

        self.epochs = epochs
        self.physics_loss = physics_loss

        self.optimizer = torch.optim.Adam(
            self.nn_solver.predictor.parameters(),
            lr=learning_rate
        )

        self.loss_history = []

    def train(self):

        for epoch in range(self.epochs):

            self.nn_solver.predictor.train()

            self.optimizer.zero_grad()

            # Physics loss
            physics_loss = self.nn_solver.physics_residual(
                self.nn_solver.collocation_points
            )

            physics_loss = physics_loss.square().sum()

            # Data loss

            out_pred = self.nn_solver.predictor(
                self.nn_solver.data_points
            )

            rho_pred = self.nn_solver.decode_rho(out_pred)

            rho_true = self.nn_solver.data_values

            data_loss = (
                torch.abs(rho_pred - rho_true)
                .square()
                .sum()
            )

            # Total loss

            loss = data_loss

            if self.physics_loss:
                loss = loss + physics_loss

            # Backpropagation

            loss.backward()

            self.optimizer.step()

            self.loss_history.append(loss.item())
            # Logging

            if epoch % 1000 == 0:

                print(
                    f"Epoch {epoch} | "
                    f"Data Loss: {data_loss:.6f} | "
                    f"Physics Loss: {physics_loss:.6f}"
                )

    def plot_loss(self):

        plt.figure(figsize=(8, 5))

        plt.plot(self.loss_history)

        plt.yscale("log")

        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.title("Training Loss")

        plt.grid(True)

        plt.show()