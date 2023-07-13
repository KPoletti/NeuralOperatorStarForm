"""
This module contains the Trainer class, which is used for training a neural network model.
It also contains utility functions for checking the visibility of data and creating animations.
"""
import logging
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from src.plottingUtils import createAnimation

import torch
import wandb
from torch import nn

# from neuralop.training.losses import LpLoss, H1Loss


logger = logging.getLogger(__name__)
logging.getLogger("wandb").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def dataVisibleCheck(data: torch.tensor, meta: dict, save: str, idx):
    """
    Creates a mp4 of a random batch of the data to check if the data is visible
    Inputs:
        data: pytorch tensor of shape (batch, time, x, y, variable
        meta: dictionary of metadata
        save: directory to save the plots
    """
    if "FNO2d" in save:
        return None
    # check if FNO3d is in save
    if "FNO3d" in save:
        data = data.permute(0, 1, 3, 4, 2)
    # reduce the data to just that batch
    data = data[idx, ...]

    # reduce the meta data to just that batch
    mass = meta["mass"][idx]
    time_data = torch.tensor([t[idx] for t in meta["time"]])
    # create an animation of the density and velocity data
    createAnimation(data, time_data, save, mass, fps=1)


class Trainer(object):
    """
    A class for training a neural network model.

    Attributes:
        model (nn.Module): The neural network model to train.
        params (dataclass): A dataclass containing the hyperparameters for training.
        device (torch.device): The device to use for training (e.g. "cpu" or "cuda").
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to use for
                                                           training.
        loss (callable): The loss function to use for training.
        plot_path (str): The directory to save the plots.
    """

    def __init__(
        self, model: nn.Module, params: dataclass, device: torch.device
    ) -> None:
        """
        Initializes a Trainer object.

        Args:
            model (nn.Module): The neural network model to train.
            params (dataclass): A dataclass containing the hyperparameters for training.
            device (torch.device): The device to use for training (e.g. "cpu" or "cuda").
        """
        self.model = model
        self.params = params
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=params.scheduler_step,
            gamma=params.scheduler_gamma,
        )
        self.loss = params.loss_fn
        self.plot_path = f"results/{self.params.path}/plots"

    def dissipativeRegularizer(
        self, data: torch.Tensor, device: torch.device
    ) -> torch.nn.MSELoss:
        """
        Computes the dissipative regularization loss for a given input tensor x.

        Args:
            x (torch.Tensor): The input tensor to compute the dissipative regularization loss for.
            device (torch.device): The device to use for computation.

        Returns:
            torch.nn.MSELoss: The dissipative regularization loss.
        """
        # TODO: Test this function

        # DISSIPATIVE REGULARIZATION
        x_diss = torch.tensor(
            self.params.sampling_fn(data.shape[0], self.params.radii, data.shape[1:]),
            dtype=torch.float,
        ).to(device)
        assert x_diss.shape == data.shape
        y_diss = self.params.target_fn(x_diss, self.params.scale_down).to(device)
        out_diss = self.model(x_diss).reshape(-1, self.params.out_dim)
        diss_loss = (
            (1 / (self.params.S**2))
            * self.params.loss_weight
            * self.params.dissloss(out_diss, y_diss.reshape(-1, self.params.out_dim))
        )  # weighted by 1 / (S**2)
        return diss_loss

    def batchLoop(self, train_loader, output_encoder=None, epoch: int = 1):
        """
        A function to loop through the batches in the training data.

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
            output_encoder (callable): The output encoder to use for decoding the output.
            epoch (int): The current epoch number.

        Returns:
            tuple: A tuple containing the training loss and the loss of the last batch.
        """
        # initialize loss
        train_loss = 0
        # loss = 0
        diss_loss = 0
        diss_loss_l2 = 0
        # select a random batch
        rand_point = np.random.randint(0, len(train_loader))
        torch.autograd.set_detect_anomaly(True)

        for batch_idx, sample in enumerate(train_loader):
            data = sample["x"].to(self.device)
            target = sample["y"].to(self.device)
            output = self.model(data)

            if epoch == 0 and batch_idx == 0:
                logger.debug(f"Batch Data Shape: {data.shape}")
                logger.debug(f"Batch Target Shape: {target.shape}")
                logger.debug(f"Output Data Shape: {output.shape}")
                logger.debug(f"Input Meta Data: {sample['meta_x']}")
                logger.debug(f"Output Meta Data: {sample['meta_y']}")

            if batch_idx == rand_point and epoch == 53:
                savename = f"{self.plot_path}/Train_"
                idx = np.random.randint(0, data.shape[0] - 1)
                dataVisibleCheck(data, sample["meta_x"], f"{savename}input", idx)
                dataVisibleCheck(target, sample["meta_y"], f"{savename}target", idx)
                dataVisibleCheck(output, sample["meta_y"], f"{savename}output", idx)

            # decode  if there is an output encoder
            if output_encoder is not None:
                # decode the target
                target = output_encoder.decode(target)
                # decode the output
                output = output_encoder.decode(output)

            if batch_idx == rand_point and epoch == 53:
                savename = f"{self.plot_path}/Train_decoded_"
                dataVisibleCheck(data, sample["meta_x"], f"{savename}input", idx)
                dataVisibleCheck(target, sample["meta_y"], f"{savename}target", idx)
                dataVisibleCheck(output, sample["meta_y"], f"{savename}output", idx)
            # compute the loss
            loss = self.loss(output.float(), target)
            if len(loss.shape) > 1:
                logger.debug(f"Loss Shape: {loss.shape}")
                logger.debug(f"Loss Value: {loss.item()}")

            # add regularizer for MNO
            if self.params.NN == "MNO":
                diss_loss = self.dissipativeRegularizer(data, self.device)
                diss_loss_l2 += diss_loss.item()
                loss += diss_loss
            # Backpropagate the loss
            loss.backward()

            self.optimizer.step()
            train_loss += loss.item()
            self.optimizer.zero_grad(set_to_none=True)

        return train_loss, loss

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        output_encoder=None,
        sweep: bool = False,
    ) -> None:
        """
        Train the model
        Input:
            train_loader: torch.utils.data.DataLoader
            valid_loader: torch.utils.data.DataLoader
        Output:
            None
        """
        # initialize the loss function
        self.loss = self.params.loss_fn
        # define a test loss function that will be the same regardless of training
        if sweep:
            # use RMSE
            test_loss_fn = nn.MSELoss(reduction="mean")
        else:
            test_loss_fn = self.loss

        logger.debug(f"len(test_loader.dataset) = {len(test_loader.dataset)}")
        logger.debug(f"len(train_loader.dataset) = {len(train_loader.dataset)}")
        # start training
        for epoch in range(self.params.epochs):
            epoch_timer = time.time()

            # set to train mode
            self.model.train()
            train_loss, _ = self.batchLoop(train_loader, output_encoder, epoch)

            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                rand_point = np.random.randint(0, len(test_loader))
                for batch_idx, sample in enumerate(test_loader):
                    data = sample["x"].to(self.device)
                    target = sample["y"].to(self.device)

                    output = self.model(data)

                    # decode  if there is an output encoder
                    if output_encoder is not None:
                        output = output_encoder.decode(output)
                        # do not decode the test target because the test data is not encode
                    if batch_idx == rand_point and epoch == 53:
                        idx = np.random.randint(0, data.shape[0] - 1)
                        savename = f"{self.plot_path}/Test_decoded_"
                        dataVisibleCheck(
                            data, sample["meta_x"], f"{savename}input", idx
                        )
                        dataVisibleCheck(
                            target, sample["meta_y"], f"{savename}target", idx
                        )
                        dataVisibleCheck(
                            output, sample["meta_y"], f"{savename}output", idx
                        )
                    test_loss += test_loss_fn(output, target).item()

            test_loss /= len(test_loader.dataset)
            train_loss /= len(train_loader.dataset)
            self.scheduler.step()

            epoch_timer = time.time() - epoch_timer
            logger.info(
                "Epoch: %d/%d \t Took: %.2f \t Training Loss: %.6f \t Test Loss: %.6f",
                epoch + 1,
                self.params.epochs,
                epoch_timer,
                train_loss,
                test_loss,
            )

            print(
                f"Epoch: {epoch+1}/{self.params.epochs} \t"
                f"Took: {epoch_timer:.2f} \t"
                f"Training Loss: {train_loss:.6f} \t"
                f"Test Loss: {test_loss:.6f}"
            )
            wandb.log({"Test-Loss": test_loss, "Train-Loss": train_loss})
        try:
            if self.params.saveNeuralNetwork:
                torch.save(
                    self.model, f"results/{self.params.path}/models/{self.params.NN}"
                )
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def plot(self, time_data, prediction, in_data, truth, roll, mass, savename=""):
        """
        Plot the prediction
        Input:
            prediction: torch.tensor
            input: torch.tensor
            truth: torch.tensor

            savename: str
        Output:
            None
        """
        logging.getLogger("matplotlib").setLevel(logging.WARNING)

        ############ RMSE ############
        num_dims = len(prediction.shape)
        num_samples = prediction.shape[0]
        ind = -1
        if num_dims == 5:
            # find which dimension is dN//2
            for i in range(1, len(prediction.shape)):
                if prediction.shape[i] == self.params.d:
                    ind_dim = i
                if prediction.shape[i] == self.params.S:
                    ind_grid = i
                if prediction.shape[i] == self.params.dN // 2:
                    ind_tim = i
            ind_grid -= 1

            # select a random int between 0 and
            ind = 0
            gif_save = f"{self.plot_path}/All_valid"
            # create an animation of the density and velocity data
            plot_input = in_data.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)
            plot_input = plot_input.flatten(0, 1)
            plot_truth = truth.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)
            plot_truth = plot_truth.flatten(0, 1)
            plot_prediction = prediction.permute(
                0, ind_tim, ind_grid, ind_grid + 1, ind_dim
            )
            plot_prediction = plot_prediction.flatten(0, 1)
            plot_roll = roll.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)
            plot_roll = plot_roll.flatten(0, 1)

            createAnimation(plot_input, time_data, f"{gif_save}_in", mass, fps=5)
            createAnimation(plot_truth, time_data, f"{gif_save}_tru", mass, fps=5)
            createAnimation(plot_prediction, time_data, f"{gif_save}_out", mass, fps=5)
            createAnimation(plot_roll, time_data, f"{gif_save}_roll", mass, fps=5)

            # reduce the data to only that indices for ind_dim and ind_tim
            prediction = prediction.index_select(ind_dim, torch.tensor(ind))
            prediction = prediction.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)
            prediction = prediction.squeeze().flatten(0, 1)
            # prediction = prediction.index_select(ind_tim, torch.tensor(t_ind)).squeeze()

            truth = truth.index_select(ind_dim, torch.tensor(ind))
            # truth = truth.index_select(ind_tim, torch.tensor(t_ind)).squeeze()
            truth = truth.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)
            truth = truth.squeeze().flatten(0, 1)

            in_data = in_data.index_select(ind_dim, torch.tensor(ind))
            in_data = in_data.permute(0, ind_tim, ind_grid, ind_grid + 1, ind_dim)
            in_data = in_data.squeeze().flatten(0, 1)
            # input = input.index_select(ind_tim, torch.tensor(t_ind)).squeeze()

            # and (n,x,y,t) flatten to (t*n, x, y)
            num_dims = len(prediction.shape)
        elif num_dims == 4:
            prediction = prediction.squeeze().flatten(0, 1)
            truth = truth.squeeze().flatten(0, 1)
            in_data = in_data.squeeze().flatten(0, 1)
            num_dims = len(prediction.shape)

        dims_to_rmse = tuple(range(-num_dims + 1, 0))

        # compute RMSE
        rmse = torch.sqrt(torch.mean((prediction - truth) ** 2, dim=dims_to_rmse))
        # normalize RMSE
        rmse /= torch.sqrt(torch.mean(truth**2, dim=dims_to_rmse))
        # compute RMSE for each time step
        relative_rmse = torch.sqrt(torch.mean((in_data - truth) ** 2, dim=dims_to_rmse))
        # normalize RMSE
        relative_rmse /= torch.sqrt(torch.mean(truth**2, dim=dims_to_rmse))

        # set the style
        sns.set_style("darkgrid")
        # plot RMSE for each time step
        fig, axis = plt.subplots(1, 1, figsize=(10, 10))
        axis.plot(
            time_data[: 3 * num_samples],
            rmse[: 3 * num_samples],
            ".",
            label="Neural RMSE",
        )
        axis.plot(
            time_data[: 3 * num_samples],
            relative_rmse[: 3 * num_samples],
            ".",
            label="RMSE between time steps",
        )
        axis.set_xlabel("Time (Myr)")
        axis.set_ylabel("Normalized  RMSE")
        axis.set_title(f"RMSE Over Time for index {ind}, t_0 {time_data[0]}")
        axis.legend()
        img = wandb.Image(fig)
        wandb.log({"RMSE": img})
        if len(savename) > 0:
            fig.savefig(f"{self.plot_path}/{savename}_RMSE.png")
        plt.close(fig)
        plt.close()

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        input_encoder=None,
        output_encoder=None,
        savename: str = "",
    ) -> None:
        """
        Evaluates the model on the given data_loader and saves the results to a .mat file.
        Args:
            data_loader (torch.utils.data.DataLoader): The data loader to evaluate the model on.
            input_encoder (optional): The input encoder to use. Defaults to None.
            output_encoder (optional): The output encoder to use. Defaults to None.
            savename (str, optional): The name to use when saving the results to a .mat file.
                                      Defaults to "".
        Returns:
            None
        """
        # TODO: TEST THIS FUNCTION
        self.model.eval()
        # create a tensor to store the prediction
        num_samples = len(data_loader.dataset)
        for batchidx, sample in enumerate(data_loader):
            data, target = sample["x"].to(self.device), sample["y"].to(self.device)
            lentime = len(sample["meta_x"]["time"])
            break

        size = [num_samples] + [d for d in data.shape if d != 1]
        pred = torch.zeros(size)
        roll = torch.zeros(size)
        in_data = torch.zeros(size)
        truth = torch.zeros(size)
        test_loss = 0
        re_loss = 0
        rand_point = np.random.randint(0, len(data_loader))
        mass = "null"
        mass_list = []
        time_data = torch.zeros(num_samples, lentime)
        with torch.no_grad():
            for batchidx, sample in enumerate(data_loader):
                data, target = sample["x"].to(self.device), sample["y"].to(self.device)
                meta_data = sample["meta_x"]
                cur_mass = meta_data["mass"][0]
                time_data[batchidx] = torch.tensor(sample["meta_y"]["time"][1:])
                # apply the input encoder
                output = self.model(data)
                # apply the model to previous output
                if batchidx == 0 or cur_mass != mass:
                    roll_batch = output.clone()
                    logging.info(f"Updating mass from M{mass} to M{cur_mass}")
                    mass = cur_mass
                elif batchidx > 0:
                    if input_encoder is not None:
                        logger.debug("Encoding reData")
                        roll_batch = input_encoder.encode(roll_batch)
                    # apply the model to previous output
                    roll_batch = self.model(roll_batch)

                if batchidx == rand_point:
                    save_plot = f"{self.plot_path}/Valid_"
                    idx = 0
                    dataVisibleCheck(data, sample["meta_x"], f"{save_plot}input", idx)
                    dataVisibleCheck(
                        target, sample["meta_y"], f"{save_plot}target", idx
                    )
                    dataVisibleCheck(
                        output, sample["meta_y"], f"{save_plot}output", idx
                    )
                    dataVisibleCheck(
                        roll_batch, sample["meta_y"], f"{save_plot}rolling", idx
                    )

                # decode  if there is an output encoder
                if output_encoder is not None:
                    logger.debug("Decoding data")
                    # decode the output
                    output = output_encoder.decode(output)
                    # decode the multiple applications of the model
                    roll_batch = output_encoder.decode(roll_batch)
                # compute the loss
                re_loss += self.loss(roll_batch, target).item()
                test_loss += self.loss(output, target).item()

                if input_encoder is not None:
                    data = input_encoder.decode(data)

                # assign the values to the tensors
                pred[batchidx] = output
                in_data[batchidx] = data
                truth[batchidx] = target
                roll[batchidx] = roll_batch

                mass_list = mass_list + [cur_mass] * lentime
            # normalize test loss
            test_loss /= len(data_loader.dataset)
            re_loss /= len(data_loader.dataset)
            print(f"Length of Valid Data: {len(data_loader.dataset)}")
            # print the final loss
            num_trained = self.params.N * self.params.split[0]
            print(f"Final Loss for {int(num_trained)}: {test_loss} ")
            wandb.log({"Valid Loss": test_loss, "Reapply Loss": re_loss})
        print(
            f"Saving data as {os.getcwd()} results/{self.params.path}/data/{savename}.mat"
        )
        # flatten the time for plotting
        time_data = time_data.flatten()

        if len(savename) > 0:
            print(
                f"Saving data as {os.getcwd()} results/{self.params.path}/data/{savename}.mat"
            )
            scipy.io.savemat(
                f"results/{self.params.path}/data/{savename}.mat",
                mdict={
                    "input": in_data.cpu().numpy(),
                    "pred": pred.cpu().numpy(),
                    "target": truth.cpu().numpy(),
                    "rePred": roll.cpu().numpy(),
                },
            )
        if self.params.doPlot:
            self.plot(
                time_data=time_data,
                prediction=pred,
                in_data=in_data,
                truth=truth,
                roll=roll,
                mass=mass_list,
                savename=savename,
            )
