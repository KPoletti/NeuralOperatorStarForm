import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO3d, FNO2d, FNO
from neuralop.training.losses import LpLoss, H1Loss
from dataclasses import dataclass
import numpy as np
import pandas as pd
import input as param
import logging
import scipy

logger = logging.getLogger(__name__)

# grab the ending of the density file with .pt
densityProfile = param.TRAIN_PATH.split("density")[-1].split(".")[0]

# create output folder
path = f"SF_{param.NN}_V100_ep{param.epochs}_m{param.modes}_w{param.width}_S{param.S}{densityProfile}"
if not os.path.exists(f"NeuralOperatorStarForm/results/{path}/logs"):
    os.makedirs(f"NeuralOperatorStarForm/results/{path}/logs")
    os.makedirs(f"NeuralOperatorStarForm/results/{path}/models")
    os.makedirs(f"NeuralOperatorStarForm/results/{path}/plots")
    os.makedirs(f"NeuralOperatorStarForm/results/{path}/data")
logger.setLevel(param.level)
fileHandler = logging.FileHandler(
    f"NeuralOperatorStarForm/results/{path}/logs/network.log", mode="w"
)
formatter = logging.Formatter(
    "%(asctime)s :: %(funcName)s :: %(levelname)s :: %(message)s"
)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

# define the network architecture


def initializeNetwork(params: dataclass) -> nn.Module:
    """
    Initialize the model
    Input:
        params: input parameters from the input file
    Output:
        model: torch.nn.Module
    """
    models = {"FNO3d": FNO3d, "MNO": FNO2d, "FNO": FNO}
    # TODO: allow MLP to be input parameter
    if params.NN == "FNO3d":
        model = FNO3d(
            n_modes_depth=params.modes,
            n_modes_width=params.modes,
            n_modes_height=2,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0},
        )
    elif params.NN == "MNO":
        model = FNO2d(
            n_modes_height=params.modes,
            n_modes_width=params.modes,
            hidden_channels=params.width,
            in_channels=params.input_channels,
            out_channels=params.output_channels,
            use_mlp=True,
            mlp={"expansion": 0.5, "dropout": 0},
        )
    logger.debug(model)
    return model


class Trainer(object):
    def __init__(
        self, model: nn.Module, params: dataclass, device: torch.device
    ) -> None:
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

    def dissipativeRegularizer(
        self, x: torch.Tensor, device: torch.device
    ) -> torch.nn.MSELoss:
        # TODO: Test this function

        # DISSIPATIVE REGULARIZATION
        x_diss = torch.tensor(
            self.params.sampling_fn(x.shape[0], self.params.radii, x.shape[1:]),
            dtype=torch.float,
        ).to(device)
        assert x_diss.shape == x.shape
        y_diss = self.params.target_fn(x_diss, self.params.scale_down).to(device)
        out_diss = self.model(x_diss).reshape(-1, self.params.out_dim)
        diss_loss = (
            (1 / (self.params.S**2))
            * self.params.loss_weight
            * self.params.dissloss(out_diss, y_diss.reshape(-1, self.params.out_dim))
        )  # weighted by 1 / (S**2)
        return diss_loss

    def batchLoop(self, train_loader, output_encoder=None, regularizer=None):
        train_loss = 0
        loss = 0
        for batch_idx, sample in enumerate(train_loader):
            data, target = sample["x"].to(self.device), sample["y"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data).view(
                self.params.batch_size,
                self.params.S,
                self.params.S,
                self.params.T,
            )

            # decode  if there is an output encoder
            if output_encoder:
                # decode the target
                target = output_encoder.decode(target)
                # decode the output
                output = output_encoder.decode(output)

            # compute the loss
            loss = self.loss(
                output.view(self.params.batch_size, -1),
                target.view(self.params.batch_size, -1),
            )
            # add regularizer for MNO
            if regularizer:
                loss += self.dissipativeRegularizer(data, self.device)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss, loss

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        output_encoder = None,
        regularizer=None,
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

        # initialize the regularizer
        # start training
        for epoch in range(self.params.epochs):
            self.model.train()
            # for batch_idx, sample in enumerate(train_loader):
            #     data, target = sample["x"].to(self.device), sample["y"].to(self.device)
            #     self.optimizer.zero_grad()
            #     output = self.model(data).view(
            #         self.params.batch_size,
            #         self.params.S,
            #         self.params.S,
            #         self.params.T,
            #     )
            #     # decode  if there is an output encoder
            #     if output_encoder:
            #         # decode the target
            #         target = output_encoder.decode(target)
            #         # decode the output
            #         output = output_encoder.decode(output)
            #     # compute the loss
            #     loss = self.loss(
            #         output.view(self.params.batch_size, -1),
            #         target.view(self.params.batch_size, -1),
            #     )
            #     # add regularizer for MNO
            #     if regularizer:
            #         loss += self.dissipativeRegularizer(data, self.device)
            #     loss.backward()
            #     self.optimizer.step()
            #     train_loss += loss.item()
            train_loss, loss = self.batchLoop(train_loader, output_encoder, regularizer)
            self.scheduler.step()

            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for sample in test_loader:
                    data, target = sample["x"].to(self.device), sample["y"].to(
                        self.device
                    )
                    output = self.model(data).view(
                        self.params.batch_size,
                        self.params.S,
                        self.params.S,
                        self.params.T,
                    )
                    # decode  if there is an output encoder
                    if output_encoder:
                        output = output_encoder.decode(output)
                        # do not decode the target because the test data is not encoded
                    test_loss += self.loss(
                        output.view(self.params.batch_size, -1),
                        target.view(self.params.batch_size, -1),
                    ).item()
            test_loss /= len(test_loader.dataset)
            train_loss /= len(train_loader.dataset)
            logger.info(
                f"Epoch: {epoch+1}/{self.params.epochs} \t"
                f"Training Loss: {train_loss:.6f} \t"
                f"Test Loss: {test_loss:.6f}"
            )
        if self.params.save_model:
            torch.save(self.model, f"NeuralOperatorStarForm/results/{path}/models/{self.params.NN}")
    def densityRMSE(self, prediction, truth,):
        """
        Calculate the RMSE as a function of density
        Input:
            prediction: torch.tensor
            input: torch.tensor
            truth: torch.tensor
        Output:
            bins: np.array
            mean_RMSE: np.array
            std_RMSE: np.array
        """
        # create bins of the data
        numBins = 100
        # create number of time steps to analyze
        numSteps = 20
        # create array to store the RMSE
        RMSE_array = np.zeros((numSteps, numBins))
        # create a list of random time steps
        randInd = np.random.randint(0, prediction.shape[0], numSteps)
        # create bins of the data
        _, bins = np.histogram(truth[ind, :, :, 0], bins=numBins)
        for j, ind in enumerate(randInd):
            # create a mask for data
            mask = np.digitize(truth[ind, :, :, 0], bins)
            # loop over the bins
            for i in range(numBins):
                truthBin = truth[ind, :, :, 0][mask == i]
                predictionBin = prediction[ind, :, :, 0][mask == i]
                # compute RMSE
                RMSE_array[j, i] = np.sqrt(np.mean((predictionBin - truthBin) ** 2))
        # compute the mean and std of the RMSE
        mean_RMSE = np.mean(RMSE_array, axis=0)
        std_RMSE = np.std(RMSE_array, axis=0)
        return bins, mean_RMSE, std_RMSE



    def plot(self, timeData, prediction, input, truth, savename=""):
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
        import matplotlib.pyplot as plt
        import seaborn as sns
        # set the style
        sns.set_style("darkgrid")
        ############ RMSE ############
        # compute RMSE
        rmse = torch.sqrt(torch.mean((prediction - truth) ** 2, dim=(1, 2, 3)))
        # normalize RMSE
        rmse /= torch.sqrt(torch.mean(truth ** 2, dim=(1, 2, 3)))
        # compute RMSE for each time step
        relativeRMSE = torch.sqrt(torch.mean((input - truth) ** 2, dim=(1, 2, 3)))
        # normalize RMSE
        relativeRMSE /= torch.sqrt(torch.mean(truth ** 2, dim=(1, 2, 3)))
        time = timeData["t"]["time"].values
        # plot RMSE for each time step
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(time, rmse, label="Neural RMSE")
        ax.plot(time, relativeRMSE, label="RMSE between time steps")
        ax.set_xlabel("Time (Myr)")
        ax.set_ylabel("Normalized  RMSE")
        ax.set_title("RMSE Over Time")
        ax.legend()
        fig.savefig(f"NeuralOperatorStarForm/results/{path}/plots/{savename}_RMSE.png")
        plt.close(fig)
        ############ RMSE Vs Density ############
        # compute the RMSE as a function of density
        predBins, predMean, predSTD = self.densityRMSE(prediction, truth)
        inputBins, inputMean, inputSTD = self.densityRMSE(input, truth)
        # plot the RMSE as a function of density
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(predBins[:-1], predMean, label="Neural RMSE")
        ax.fill_between( predBins[:-1], predMean - predSTD, predMean + predSTD, alpha=0.2)
        ax.plot(inputBins[:-1], inputMean, label="RMSE between time steps")
        ax.fill_between( inputBins[:-1], inputMean - inputSTD, inputMean + inputSTD, alpha=0.2)
        ax.set_xlabel(r"Column Density $  \left[\frac{g}{cm^2} \right]$")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.set_title("RMSE as a function of density")
        fig.savefig(f"NeuralOperatorStarForm/results/{path}/plots/{savename}_RMSE_vs_density.png")
        plt.close(fig)
    



    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        timeData: pd.dataframe,
        input_encoder=None,
        output_encoder=None,
        savename: str = "",
    ) -> None:
        """
        Evaluate the model
        Input:
            data_loader: torch.utils.data.DataLoader
        Output:
            None
        """
        # TODO: TEST THIS FUNCTION
        self.model.eval()

        num_samples = len(data_loader) * self.params.batch_size
        pred = torch.zeros((num_samples, self.params.S, self.params.S, self.params.T))
        rePred = torch.zeros((num_samples, self.params.S, self.params.S, self.params.T))
        input = torch.zeros((num_samples, self.params.S, self.params.S, self.params.T))
        truth = torch.zeros((num_samples, self.params.S, self.params.S, self.params.T))
        test_loss = 0
        with torch.no_grad():
            for batchidx, sample in enumerate(data_loader):
                data, target = sample["x"].to(self.device), sample["y"].to(self.device)
                if batchidx > 0:
                    # apply the model to previous output
                    reData = self.model(output).view(
                        self.params.batch_size,
                        self.params.S,
                        self.params.S,
                        self.params.T,
                    )
                output = self.model(data).view(
                    self.params.batch_size,
                    self.params.S,
                    self.params.S,
                    self.params.T,
                )
                # decode  if there is an output encoder
                if output_encoder:
                    # decode the target
                    target = output_encoder.decode(target)
                    # decode the output
                    output = output_encoder.decode(output)
                    # decode the input
                    data = input_encoder.decode(data)
                test_loss += self.loss(
                    output.view(self.params.batch_size, -1),
                    target.view(self.params.batch_size, -1),
                ).item()
                if input_encoder:
                    data = input_encoder.decode(data)
                # assign the values
                index = batchidx * self.params.batch_size
                pred[index : index + self.params.batch_size] = output
                input[index : index + self.params.batch_size] = data
                truth[index : index + self.params.batch_size] = target
                if batchidx > 0:
                    rePred[index : index + self.params.batch_size] = reData
                else:
                    rePred[index : index + self.params.batch_size] = output

        if len(savename) > 0:
            scipy.io.savemat(
                f"NeuralOperatorStarForm/results/{path}/models/{savename}.mat",
                mdict={
                    "input": input.cpu().numpy(),
                    "pred": pred.cpu().numpy(),
                    "target": truth.cpu().numpy(),
                },
            )
        if self.params.doPlot:
            self.plot(pred, input, truth, savename)
    