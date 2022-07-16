from flax import linen as nn
import jax.numpy as jnp


class UNet1D(nn.Module):
    features: tuple
    n_components: int
    dw: int

    def setup(self):
        self.pool_kernel = self.features[0] - int(self.features[0] / 4) + 1

        # DownSample L1
        self.conv1 = nn.Conv(
            2 * self.features[1] * self.n_components,
            kernel_size=[2 * self.dw + 1],
            padding=[(self.dw, self.dw)],
            use_bias=False,
        )
        self.conv2 = nn.Conv(
            self.features[1] * self.n_components,
            kernel_size=[1],
            padding=[(0, 0)],
            use_bias=False,
        )

        # Concatenation
        self.conv21 = nn.Conv(
            self.features[1] * self.n_components,
            kernel_size=[1],
            padding=[(0, 0)],
            use_bias=False,
        )
        self.conv22 = nn.Conv(
            self.features[1] * self.n_components,
            kernel_size=[1],
            padding=[(0, 0)],
            use_bias=False,
        )
        self.conv23 = nn.Conv(
            self.features[1] * self.n_components,
            kernel_size=[1],
            padding=[(0, 0)],
            use_bias=False,
        )

        # DOWNSAMPLES
        self.conv3 = nn.Conv(
            self.features[1] * self.n_components,
            kernel_size=[1],
            padding=[(0, 0)],
            use_bias=False,
        )

        # UPSAMPLE
        self.conv2T = nn.ConvTranspose(
            self.features[1],
            kernel_size=[4],
            strides=[4],
            padding="SAME",
            use_bias=False,
        )
        self.convHR1 = nn.Conv(
            2 * self.features[1] * self.n_components,
            kernel_size=[2 * self.dw + 1],
            padding=[(self.dw, self.dw)],
            use_bias=False,
        )
        self.convHR2 = nn.Conv(
            self.features[1] * self.n_components,
            kernel_size=[1],
            padding=[(0, 0)],
            use_bias=False,
        )
        self.convHR21 = nn.Conv(
            self.features[1] * self.n_components,
            kernel_size=[1],
            padding=[(0, 0)],
            use_bias=False,
        )
        self.convHR22 = nn.Conv(
            self.features[1] * self.n_components,
            kernel_size=[1],
            padding=[(0, 0)],
            use_bias=False,
        )
        self.convHR23 = nn.Conv(
            self.features[1] * self.n_components,
            kernel_size=[1],
            padding=[(0, 0)],
            use_bias=False,
        )
        self.convHR3 = nn.Conv(
            self.features[1], kernel_size=[1], padding=[(0, 0)], use_bias=False
        )

    def __call__(self, x, training):

        # ======================
        # POOLING
        # ======================
        # Bx200x3 -> Bx50x3
        x_lr = nn.avg_pool(inputs=x, window_shape=(self.pool_kernel,))

        # ===========================
        # LOW RES
        # ===========================

        x_lr = self.conv1(x_lr)
        x_lr = nn.relu(x_lr)
        x_lr = self.conv2(x_lr)

        # CONCATENTATION LAYER
        x_lr = jnp.concatenate(
            [self.conv21(x_lr), self.conv22(x_lr) * self.conv23(x_lr)], axis=2
        )

        x_lr = self.conv3(x_lr)

        # UPSAMPLE
        x_lr = self.conv2T(x_lr)

        # ===========================
        # HIGH RES
        # ===========================

        x_hr = self.convHR1(x)
        x_hr = nn.relu(x_hr)
        x_hr = self.convHR2(x_hr)

        # CONCATENTATION LAYER
        x_hr = jnp.concatenate(
            [self.convHR21(x_hr), self.convHR22(x_hr) * self.convHR23(x_hr)], axis=2
        )

        x_hr = self.convHR3(x_hr)

        x = x_lr + x_hr

        return x
