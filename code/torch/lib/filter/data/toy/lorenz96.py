import torch
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
from typing import NamedTuple
import matplotlib.pyplot as plt
from dataclasses import dataclass


def lorenz96_gradient_step(x: np.ndarray, t: int, F: float, D: float) -> np.ndarray:
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
    dx = np.zeros(D)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(D):
        dx[i] = (x[(i + 1) % D] - x[i - 2]) * x[i - 1] - x[i] + F
    return dx


class Simulations(NamedTuple):
    x0: np.ndarray
    x: np.ndarray
    t: np.ndarray


class Catalog(NamedTuple):
    x: np.ndarray
    analogs: np.ndarray
    successors: np.ndarray


class Lorenz96(NamedTuple):

    name: int = "Lorenz_96"

    # LORENZ 96 PARAMS
    F: int = 8  # Forcing term
    D: int = 40  # number of variables (states)

    # TIME STEPS
    x0_perturb: float = 0.01  # initial perturbation for the first variable
    dt: float = 0.01  # sampling time

    xd2_perturb: float = 0.01
    time_buffer: float = 1e-6  # for numerical errors on the boundary

    # NUM DATA PARAMS
    n_train: int = 100  # number of train steps
    n_test: int = 100  # size of true state & noisy obs
    n_steps_init: int = 5  # number of initial steps to be in attractor space

    # TRUE STATE PARAMS
    t0: float = 0.01  # t0 for the true state
    t1: float = 100.0  # t1 for the true state

    # NOISY PARAMS
    seed: int = 123  # seed for the random noise
    sigma: float = 0.0  # variance of model error


def create_initial_states(model: Lorenz96) -> np.ndarray:
    """Generate initial states

    Returns:
        x0 (array) : size=(self.J)
    """
    x0 = model.F * np.ones(model.D)

    # create a perturbation in the middle
    if model.xd2_perturb is not None:
        idx = np.int64(np.around(model.D / 2))
        x0[idx] += model.xd2_perturb

    # Add small perturbation to the first variable
    if model.x0_perturb is not None:
        x0[0] += model.x0_perturb

    return x0


def generate_simulations_pre(model: Lorenz96) -> Simulations:

    # define function
    fn = lorenz96_gradient_step

    # define initial condition
    x0 = create_initial_states(model)

    # define time steps
    dt = model.dt
    t0 = 0
    t1 = model.n_steps_init + model.time_buffer

    time_steps = np.arange(t0, t1, dt)

    # other arguments
    args = (model.F, model.D)

    # solve ODE
    S = odeint(fn, x0, time_steps, args=args)

    # save simulations
    sims = Simulations(x0=x0, x=S, t=time_steps)

    return sims


def gen_simulations_post(model: Lorenz96) -> Simulations:

    # define function
    fn = lorenz96_gradient_step

    # generate simulations, pre-attractor
    sims = generate_simulations_pre(model)
    # new init based on one trajectory (not the boundary)
    end_idx = sims.x.shape[0] - 1
    x0 = sims.x[end_idx, :]

    # define time steps
    dt = model.dt
    t0 = 0.01
    t1 = model.n_test + model.time_buffer

    time_steps = np.arange(t0, t1, dt)

    # other arguments
    args = (model.F, model.D)

    # solve ODE
    S = odeint(fn, x0, time_steps, args=args)

    # save simulations
    sims = Simulations(x0=x0, x=S, t=time_steps)

    return sims


def generate_test_data(model: Lorenz96) -> Simulations:

    # generate simulations of true state
    sims = gen_simulations_post(model)

    # create the rest
    S = sims.x
    x0 = sims.x0
    n_test = model.n_test

    t_xt = np.arange(0, n_test, 1)

    # create vectors
    time = t_xt * model.dt
    x_values = S[t_xt, :]

    sims_test = Simulations(x0=sims.x0, x=x_values, t=time)

    return sims_test


def generate_noisy_obs(sims: Simulations, model: Lorenz96):

    # unroll params
    n_dims = model.D

    # define function
    fn = lorenz96_gradient_step

    # new init based on one trajectory (not the boundary)
    end_idx = sims.x.shape[0] - 1
    x0 = sims.x[end_idx, :]

    # define time steps
    dt = model.dt
    t0 = 0.01
    t1 = model.n_train + model.time_buffer

    time_steps = np.arange(t0, t1, dt)

    # other arguments
    args = (model.F, model.D)

    # solve ODE
    S = odeint(fn, x0, time_steps, args=args)
    # generate noise vector
    if model.sigma is not None:
        n_time_steps = S.shape[0]
        rng = np.random.RandomState(model.seed)
        mean = np.zeros(model.D)
        cov = np.eye(model.D, model.D)
        cov *= model.sigma
        eta = rng.multivariate_normal(mean, cov, n_time_steps)
        S += eta

    # save simulations
    sims = Simulations(x0=x0, x=S, t=time_steps)

    return sims


def generate_catalog(sims: Simulations):
    return Catalog(x=sims.x, analogs=sims.x[0:-1], successors=sims.x[1:])
