import unittest

import torch as t

from lgssm import LinearGaussianStateSpaceModel


class TestLGSSM(unittest.TestCase):
    def test_lgssm(self):
        t.manual_seed(0)
        dt = 1 / 10
        prior_mean = t.zeros(4, dtype=t.float64)
        prior_covariance = t.eye(4, dtype=t.float64)
        transition_matrix = t.tensor(
            [  # 4x4
                # x, y, vx, vy
                [1.0, 0.0, 0.0, 0.0],  # x
                [0.0, 1.0, 0.0, 0.0],  # y
                [dt, 0.0, 1.0, 0.0],  # vx
                [0.0, dt, 0.0, 1.0],  # vy
            ],
            dtype=t.float64,
        )
        transition_covariance = 1e-8 * t.eye(4, dtype=t.float64)
        observation_matrix = t.tensor(
            [
                # x, y
                [1, 0],  # x
                [0, 1],  # y
                [0, 0],  # vx
                [0, 0],  # vy
            ],
            dtype=t.float64,
        )
        observation_covariance = 0.01 * t.eye(2, dtype=t.float64)

        lgssm = LinearGaussianStateSpaceModel(
            prior_mean,
            prior_covariance,
            transition_matrix,
            transition_covariance,
            observation_matrix,
            observation_covariance,
        )
        n_steps = 32
        x, z = lgssm.sample(n_steps)

        logpx_expected = 34.75622086749678  # Computed with pykalman

        logpx_actual, steps = lgssm.log_prob(x.unsqueeze(1))
        self.assertAlmostEqual(logpx_expected, logpx_actual.item(), places=4)
        self.assertEqual(n_steps, len(steps))
