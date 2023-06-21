# Datasets

## 2-D Incompressible Navier-Stokes

Researchers at Caltech simulated Navier stokes on

- [KF-Re100](<https://caltech-pde-data.s3.us-west-2.amazonaws.com/KFvorticity_Re100_N50_T500.npy>) - never used this dataset
- [Navier-Stokes](<https://caltech-pde-data.s3.us-west-2.amazonaws.com/ns_V1e-3_N5000_T50.mat>) - 2-D incompressible Navier-Stokes with viscosity 0.001, with shape of
 (# of trajectories, grid_x, gird_y, timestep per trajectory)= (5000, 64, 64, 51)

## CATS MHD Sim

From CATS MHD Simulations, shape of (# of trajectories, grid_x, gird_y, timestep per trajectory)=[2970, 256, 256, 2]

## MHD Turbulence

Simulation of turbulent MHD with shape of (# of trajectories, grid_x, gird_y, timesteps per trajectory). Each dataset has multiple projections of the same simulation in different directions.

| Dataset | Shape | filename |
| ------- | ----- | -------- |
| Mu2     | [874, 800, 800, 2]   | `density_mu2_dN10`|
| Mu8     | [875, 800, 800, 2]   | `density_mu8_dN10`|
| Mu32    | [954, 800, 800, 2]   | `density_mu32_dN10`|
| Mu0     | [774, 800, 800, 2]   | `density_mu0_dN10`|

## Gravitational Collapse

[GIZMO](http://www.tapir.caltech.edu/~phopkins/Site/GIZMO.html) simulation of spherically symmetric gas cloud with different virial parameters by varying the initial mass.   Filenames are of the form

~~~text
f"Grav_M{mass in solar masses}_dN{timesteps per trajectory}_dt{size of timestep in kyr}.pt"
~~~

If the file ends with `multiData`, then data is for density and velocity in the x and y directions. Otherwise, the file contains only density. Most files have the shape of [30, 800, 800, 3, dN], where dN is the number of timesteps per trajectory. The exceptions are files that start with `Grav_MALL` which have all the data in one file. They are the shape of [902, g, g, 3, dN], where g is the grid size and is either 800 or 200.
