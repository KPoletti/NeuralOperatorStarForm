import torch
import torch.nn.functional as F

from neuralop.losses.finite_diff import central_diff_2d


class ContinuityLoss(object):
    """
    Computes continuity loss for the continuity equation.
    """

    def __init__(self, loss=F.mse_loss, timestep=8.29, length=1.25) -> None:
        self.loss = loss
        self.timestep = timestep
        self.length = length
        if not isinstance(self.length, (tuple, list)):
            self.length = [self.length] * 2

    def eqn(self, rho_old, rho_new, u_old):
        """
        rho_new: predicted density (batch_size,t,nx,ny)
        u_old: predicted old velocity (batch_size,d,nx,ny)
        rho_old: given density (batch_size,t,nx,ny)
        """
        # remove extra channel dimensions
        rho_new = rho_new.squeeze(1)
        rho_old = rho_old.squeeze(1)
        u_x = u_old[:, 0]
        u_y = u_old[:, 1]
        # shapes
        _, nx, ny = rho_new.shape

        dx = self.length[0] / nx
        dy = self.length[1] / ny

        drhodx, drhody = central_diff_2d(
            rho_old, [dx, dy], fix_x_bnd=True, fix_y_bnd=True
        )
        du_xdx, _ = central_diff_2d(
            u_x, [dx, dy], fix_x_bnd=True, fix_y_bnd=True
        )
        _, du_ydy = central_diff_2d(
            u_y, [dx, dy], fix_x_bnd=True, fix_y_bnd=True
        )
        drhodt = -(rho_new - rho_old) / self.timestep
        div = drhodx * u_x + drhody * u_y + rho_old * (du_xdx + du_ydy)
        return self.loss(drhodt, div)

    def __call__(self, rho_new, u_old, rho_old):
        return self.eqn(rho_new, u_old, rho_old)
