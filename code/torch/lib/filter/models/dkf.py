# TODO: add DKF with flattener (https://github.com/hmsandager/Normalizing-flow-and-deep-kalman-filter/blob/main/dkf.py)
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from typing import Callable
from ml4da._src.models.layers.rnn import get_default_rnn_layer


class DeepKalmanFilter(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Kalman Filter
    """

    def __init__(
        self,
        emitter_fn: Callable,
        transition_fn: Callable,
        combiner_fn: Callable,
        input_dim: int = 1,
        z_dim: int = 10,
        emission_dim: int = 30,
        transition_dim: int = 30,
        rnn_dim: int = 10,
        num_layers: int = 1,
        use_cuda: int = False,
        annealing_factor: int = 1.0,
        nonlinearity: str = "relu",
        bidirectional: bool = False,
    ):
        super(DeepKalmanFilter, self).__init__()
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = emitter_fn(input_dim, z_dim, emission_dim)
        self.trans = transition_fn(z_dim, transition_dim)
        self.combiner = combiner_fn(z_dim, rnn_dim)
        # TODO manually define pytorch module
        self.rnn = get_default_rnn_layer(
            input_size=input_dim,
            hidden_size=rnn_dim,
            nonlinearity=nonlinearity,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.sigma = nn.Parameter(torch.ones(input_dim) * 0.3)
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        self.annealing_factor = annealing_factor
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(self, sequence=None):
        # get batch_size
        batch_size = len(sequence)
        # this is the number of time steps we need to process in the mini-batch
        T_max = len(sequence[0]) if isinstance(sequence, list) else sequence.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dkf", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(batch_size, self.z_0.size(0))
        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        with pyro.plate("data", batch_size):
            mus = torch.zeros((batch_size, T_max, 1))
            sigmas = torch.zeros((batch_size, T_max, 1))
            # sample the latents z and observed x's one time step at a time
            for t in range(1, T_max + 1):
                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale)
                with poutine.scale(scale=self.annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t, dist.Normal(z_loc, z_scale).to_event(1)
                    )
                # compute the probabilities that parameterize the bernoulli likelihood
                emission_mu_t = self.emitter(z_t)
                mus[:, t - 1, :] = emission_mu_t
                # the next statement instructs pyro to observe x_t according to the
                # gaussian distribution p(x_t|z_t)
                if isinstance(sequence, list):
                    pyro.sample(
                        "obs_y_%d" % t,
                        dist.Normal(loc=emission_mu_t, scale=self.sigma).to_event(1),
                        obs=None,
                    )
                else:
                    pyro.sample(
                        "obs_y_%d" % t,
                        dist.Normal(loc=emission_mu_t, scale=self.sigma).to_event(1),
                        obs=sequence[:, t - 1, :].view(-1),
                    )
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t
            return mus

    def guide(self, sequence=None):
        # get batch_size
        batch_size = len(sequence)
        # this is the number of time steps we need to process in the mini-batch
        T_max = len(sequence[0]) if isinstance(sequence, list) else sequence.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dkf", self)
        # if on gpu we need the fully broadcast view of the rnn initial state
        # to be in contiguous gpu memory
        h_0_contig = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()
        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        rnn_output, rnn_hidden_state = self.rnn(sequence, h_0_contig)

        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("data", batch_size):
            # sample the latents z one time step at a time
            for t in range(1, T_max + 1):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{:t})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                z_dist = dist.Normal(z_loc, z_scale)
                assert z_dist.event_shape == ()
                assert z_dist.batch_shape == (batch_size, self.z_q_0.size(0))
                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=self.annealing_factor):
                    # ".to_event(1)" indicates latent dimensions are independent
                    z_t = pyro.sample("z_%d" % t, z_dist.to_event(1))
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t
