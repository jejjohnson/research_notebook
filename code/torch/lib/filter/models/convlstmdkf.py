import torch
import pyro
import torch.nn as nn
from ml4da._src.models.layers.convlstm import ConvLSTM
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(
        self,
        emission_fn,
        transition_fn,
        combiner_fn,
        encoder_fn=None,
        input_channels=1,
        z_channels=16,
        emission_channels=32,
        transition_channels=32,
        rnn_channels=32,
        encoder_channels=32,
        kernel_size=3,
        height=100,
        width=100,
        pred_input_dim=5,
        encoder_layer=2,
        num_layers=2,
        rnn_dropout_rate=0.0,
        num_iafs=0,
        iaf_dim=50,
        use_cuda=False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.bottleneck = True
        self.encoder_channels = encoder_channels

        # instantiate PyTorch modules used in the model and guide below
        self.emitter = emission_fn(
            input_channels,
            z_channels,
            emission_channels,
            kernel_size,
            self.bottleneck,
            encoder_layer,
        )
        self.trans = transition_fn(
            z_channels, transition_channels, kernel_size, width, height
        )
        self.combiner = combiner_fn(z_channels, rnn_channels[0], kernel_size)
        if self.bottleneck and encoder_fn is not None:
            self.obs_encoder = encoder_fn(
                width,
                height,
                input_channels,
                self.encoder_channels,
                kernel_size,
                encoder_layer,
            )

        # Instantiate ConvLSTM
        if use_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        if self.bottleneck:
            if encoder_layer == 1:
                self.rnn_input_channels = self.encoder_channels
            else:
                self.rnn_input_channels = self.encoder_channels * (
                    2 ** (encoder_layer - 1)
                )
        self.convlstm = ConvLSTM(
            self.rnn_input_channels,
            rnn_channels,
            kernel_size,
            pred_input_dim,
            self.device,
        )

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        if self.bottleneck:
            height = height // (2**encoder_layer)
            width = width // (2**encoder_layer)
        self.z_0 = nn.Parameter(torch.zeros(z_channels, height, width))
        self.z_q_0 = nn.Parameter(torch.zeros(z_channels, height, width))
        self.h_0 = nn.Parameter(torch.zeros(rnn_channels[0], height, width))
        self.c_0 = nn.Parameter(torch.zeros(rnn_channels[0], height, width))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(
        self, mini_batch, mini_batch_reversed, mini_batch_mask, annealing_factor=1.0
    ):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)

        # register all PyTorch (sub)modules with pyro
        # this needs to happen in both the model and guide
        pyro.module("dmm", self)

        # set z_prev = z_0 to setup the recursive conditioning in p(z_t | z_{t-1})
        z_prev = self.z_0.expand(
            mini_batch.size(0), self.z_0.size(0), self.z_0.size(1), self.z_0.size(2)
        )

        # we enclose all the sample statements in the model in a plate.
        # this marks that each datapoint is conditionally independent of the others
        # Dim=-3 counted from the right -> (-3, -2, -1, 0) : (bs, dim, w, h)
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z and observed x's one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next chunk of code samples z_t ~ p(z_t | z_{t-1})
                # note that (both here and elsewhere) we use poutine.scale to take care
                # of KL annealing. we use the mask() method to deal with raggedness
                # in the observed data (i.e. different sequences in the mini-batch
                # have different lengths)

                # first compute the parameters of the diagonal gaussian distribution p(z_t | z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # then sample z_t according to dist.Normal(z_loc, z_scale)
                # note that we use the reshape method so that the univariate Normal distribution
                # is treated as a multivariate Normal distribution with a diagonal covariance.
                # TODO: Might give error because the distribution is an image. Same with other distribution based sampling.
                with poutine.scale(scale=annealing_factor):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        dist.Normal(z_loc, z_scale).to_event(dependent_event_dim),
                    )

                # compute the probabilities that parameterize the bernoulli likelihood
                emission_loc_t, emission_scale_t = self.emitter(z_t)
                # the next statement instructs pyro to observe x_t according to the
                # bernoulli distribution p(x_t|z_t)
                pyro.sample(
                    "obs_x_%d" % t,
                    dist.Normal(emission_loc_t, emission_scale_t).to_event(
                        dependent_event_dim
                    ),
                    obs=mini_batch[:, t - 1, :, :, :],
                )
                # the latent sampled at this time step will be conditioned upon
                # in the next time step so keep track of it
                z_prev = z_t

    # the guide q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(
        self, mini_batch, mini_batch_reversed, mini_batch_mask, annealing_factor=1.0
    ):

        # this is the number of time steps we need to process in the mini-batch
        T_max = mini_batch.size(1)
        # register all PyTorch (sub)modules with pyro
        pyro.module("dmm", self)

        h_0 = self.h_0.expand(
            mini_batch.size(0), self.h_0.size(0), self.h_0.size(1), self.h_0.size(2)
        )
        c_0 = self.c_0.expand(
            mini_batch.size(0), self.c_0.size(0), self.c_0.size(1), self.c_0.size(2)
        )

        # encode every observed x
        if self.bottleneck:
            batch_size = mini_batch_reversed.shape[0]
            seq_len = mini_batch_reversed.shape[1]
            enc_mini_batch_reversed = torch.zeros(
                batch_size,
                seq_len,
                self.rnn_input_channels,
                self.h_0.size(1),
                self.h_0.size(2),
            ).to(self.device)
            for t in range(seq_len):
                enc_mini_batch_reversed[:, t, :, :, :] = self.obs_encoder(
                    mini_batch_reversed[:, t, :, :, :]
                )
            mini_batch_reversed = enc_mini_batch_reversed

        # push the observed x's through the rnn;
        # rnn_output contains the hidden state at each time step
        _, rnn_output = self.convlstm(mini_batch_reversed, 0, h_0, c_0)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = reverse_sequences(rnn_output)
        # set z_prev = z_q_0 to setup the recursive conditioning in q(z_t |...)
        z_prev = self.z_q_0.expand(
            mini_batch.size(0),
            self.z_q_0.size(0),
            self.z_q_0.size(1),
            self.z_q_0.size(2),
        )

        # we enclose all the sample statements in the guide in a plate.
        # this marks that each datapoint is conditionally independent of the others.
        with pyro.plate("z_minibatch", len(mini_batch)):
            # sample the latents z one time step at a time
            # we wrap this loop in pyro.markov so that TraceEnum_ELBO can use multiple samples from the guide at each z
            for t in pyro.markov(range(1, T_max + 1)):
                # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :, :, :])

                # if we are using normalizing flows, we apply the sequence of transformations
                # parameterized by self.iafs to the base distribution defined in the previous line
                # to yield a transformed distribution that we use for q(z_t|...)
                z_dist = dist.Normal(z_loc, z_scale)

                assert z_dist.event_shape == ()
                assert z_dist.batch_shape == (
                    len(mini_batch),
                    self.z_q_0.size(0),
                    self.z_q_0.size(1),
                    self.z_q_0.size(2),
                )

                # sample z_t from the distribution z_dist
                with pyro.poutine.scale(scale=annealing_factor):
                    # when no normalizing flow used, ".to_event(1)" indicates latent dimensions are independent
                    z_t = pyro.sample("z_%d" % t, z_dist.to_event(dependent_event_dim))

                    assert z_t.shape == (
                        len(mini_batch),
                        self.z_q_0.size(0),
                        self.z_q_0.size(1),
                        self.z_q_0.size(2),
                    )
                # the latent sampled at this time step will be conditioned upon in the next time step
                # so keep track of it
                z_prev = z_t

        return z_t


def reverse_sequences(rnn_output):
    T = rnn_output.size(1)
    time_slices = torch.arange(T - 1, -1, -1, device=rnn_output.device)

    return rnn_output.index_select(1, time_slices)


def do_prediction(
    dmm, pred_batch, pred_batch_reversed, pred_batch_mask, pred_length, ground_truth
):
    # Do prediction from previous observations
    with torch.no_grad():
        # Initialization
        bsize, input_seq_len, input_channels, width, height = pred_batch.shape
        # pred_latents_loc = torch.Tensor(bsize, pred_length, input_channels, width, height)
        # pred_latents_scale = torch.Tensor(bsize, pred_length, input_channels, width, height)
        pred_observations_loc = torch.Tensor(
            bsize, pred_length, input_channels, width, height
        ).to(pred_batch.device)
        pred_observations_scale = torch.Tensor(
            bsize, pred_length, input_channels, width, height
        ).to(pred_batch.device)

        # Use guide to calculate latents from observations
        z_prev = dmm.guide(pred_batch, pred_batch_reversed, pred_batch_mask)

        # Use model to predict next latents and generate observation from them
        for i in range(0, pred_length):
            z_pred_loc, z_pred_scale = dmm.trans(z_prev)
            z_t = pyro.sample(
                "z_pred_%d" % i,
                dist.Normal(z_pred_loc, z_pred_scale).to_event(dependent_event_dim),
            )

            emission_loc_t, emission_scale_t = dmm.emitter(z_t)
            x_t = pyro.sample(
                "x_pred_%d" % i,
                dist.Normal(emission_loc_t, emission_scale_t).to_event(
                    dependent_event_dim
                ),
            )

            # Insert into tensors
            # pred_latents_loc[:, i, :, :, :] = z_pred_loc
            # pred_latents_scale[:, i, :, :, :] = z_pred_scale
            pred_observations_loc[:, i, :, :, :] = emission_loc_t
            pred_observations_scale[:, i, :, :, :] = emission_scale_t

            z_prev = z_t

        observations_loss_loc = (
            torch.sum((pred_observations_loc - ground_truth) ** 2)
            .detach()
            .cpu()
            .numpy()
        )
        observations_loss_scale = (
            torch.sum(
                (
                    pred_observations_scale
                    - torch.ones(ground_truth.shape).to(pred_batch.device)
                )
                ** 2
            )
            .detach()
            .cpu()
            .numpy()
        )

    return (
        pred_observations_loc,
        pred_observations_scale,
        observations_loss_loc,
        observations_loss_scale,
    )


def do_prediction_rep_inference(
    dmm, pred_batch_mask, pred_length, input_pred_length, ground_truth, summed=True
):
    # Do prediction from previous observations
    with torch.no_grad():
        # Initialization
        bsize, input_seq_len, input_channels, width, height = ground_truth.shape
        # pred_latents_loc = torch.Tensor(bsize, pred_length, input_channels, width, height)
        # pred_latents_scale = torch.Tensor(bsize, pred_length, input_channels, width, height)
        pred_observations_loc = torch.Tensor(
            bsize, pred_length, input_channels, width, height
        ).to(ground_truth.device)
        pred_observations_scale = torch.Tensor(
            bsize, pred_length, input_channels, width, height
        ).to(ground_truth.device)

        for i in range(pred_length):
            input_pred_start = input_seq_len - input_pred_length - pred_length + i
            if input_pred_start < 0:
                input_pred_start = 0
            input_pred_end = input_pred_start + input_pred_length
            if input_pred_end > input_seq_len:
                input_pred_end = input_seq_len
            pred_batch = ground_truth[:, input_pred_start:input_pred_end, :, :, :]
            pred_batch_reversed = reverse_sequences(pred_batch)
            assert pred_batch.shape[1] == input_pred_length

            # Use guide to calculate latents from observations
            z_prev = dmm.guide(pred_batch, pred_batch_reversed, pred_batch_mask)

            # Use model to predict next latents and generate observation from them
            z_pred_loc, z_pred_scale = dmm.trans(z_prev)
            z_t = pyro.sample(
                "z_pred_%d" % i,
                dist.Normal(z_pred_loc, z_pred_scale).to_event(dependent_event_dim),
            )

            emission_loc_t, emission_scale_t = dmm.emitter(z_t)
            x_t = pyro.sample(
                "x_pred_%d" % i,
                dist.Normal(emission_loc_t, emission_scale_t).to_event(
                    dependent_event_dim
                ),
            )

            # Insert into tensors
            # pred_latents_loc[:, i, :, :, :] = z_pred_loc
            # pred_latents_scale[:, i, :, :, :] = z_pred_scale
            pred_observations_loc[:, i, :, :, :] = emission_loc_t
            pred_observations_scale[:, i, :, :, :] = emission_scale_t

            z_prev = z_t

        if summed:
            observations_loss_loc = (
                torch.sum(
                    (
                        pred_observations_loc
                        - ground_truth[:, input_seq_len - pred_length :, :, :, :]
                    )
                    ** 2
                )
                .detach()
                .cpu()
                .numpy()
            )
            observations_loss_scale = (
                torch.sum(
                    (
                        pred_observations_scale
                        - torch.ones(
                            ground_truth[
                                :, input_seq_len - pred_length :, :, :, :
                            ].shape
                        ).to(pred_batch.device)
                    )
                    ** 2
                )
                .detach()
                .cpu()
                .numpy()
            )
        else:
            observations_loss_loc = (
                (
                    (
                        pred_observations_loc
                        - ground_truth[:, input_seq_len - pred_length :, :, :, :]
                    )
                    ** 2
                )
                .detach()
                .cpu()
                .numpy()
            )
            observations_loss_scale = (
                (
                    (
                        pred_observations_scale
                        - torch.ones(
                            ground_truth[
                                :, input_seq_len - pred_length :, :, :, :
                            ].shape
                        ).to(pred_batch.device)
                    )
                    ** 2
                )
                .detach()
                .cpu()
                .numpy()
            )

    return (
        pred_observations_loc,
        pred_observations_scale,
        observations_loss_loc,
        observations_loss_scale,
    )


if __name__ == "__main__":
    # Test
    input_channels = 1
    z_channels = 1
    emission_channels = 16
    transition_channels = 64
    rnn_channels = [64, 64]
    kernel_size = 3
    pred_length = 0
    input_length = 30
    width = 100
    height = 100

    input_tensor = torch.zeros(16, input_length, input_channels, width, height).cuda()
    input_tensor_mask = torch.ones(
        16, input_length, input_channels, width, height
    ).cuda()
    input_tensor_reversed = reverse_sequences(input_tensor).cuda()

    pred_tensor = input_tensor[:, :25, :, :, :]
    pred_tensor_mask = input_tensor_mask[:, :25, :, :, :]
    pred_tensor_reversed = reverse_sequences(pred_tensor).cuda()

    ground_truth = input_tensor[:, 25:, :, :, :]

    pred_input_dim = 5

    dmm = DMM(
        input_channels=input_channels,
        z_channels=z_channels,
        emission_channels=emission_channels,
        transition_channels=transition_channels,
        rnn_channels=rnn_channels,
        kernel_size=kernel_size,
        height=height,
        width=width,
        pred_input_dim=5,
        num_layers=2,
        rnn_dropout_rate=0.0,
        num_iafs=0,
        iaf_dim=50,
        use_cuda=True,
    )

    learning_rate = 0.01
    beta1 = 0.9
    beta2 = 0.999
    clip_norm = 10.0
    lr_decay = 1.0
    weight_decay = 0
    adam_params = {
        "lr": learning_rate,
        "betas": (beta1, beta2),
        "clip_norm": clip_norm,
        "lrd": lr_decay,
        "weight_decay": weight_decay,
    }
    adam = ClippedAdam(adam_params)

    elbo = Trace_ELBO()
    svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)
    for i in range(100):
        loss = svi.step(input_tensor, input_tensor_reversed, input_tensor_mask)
        val_nll = svi.evaluate_loss(
            input_tensor, input_tensor_reversed, input_tensor_mask
        )
        print(val_nll)
        _, _, loss_loc, loss_scale = do_prediction(
            dmm, pred_tensor, pred_tensor_reversed, pred_tensor_mask, 5, ground_truth
        )
        print(loss_loc, loss_scale)
