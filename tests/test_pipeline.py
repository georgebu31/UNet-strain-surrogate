import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exciton_solver_fixed import bernoulli, gaussian_G, solve_exciton_fixed
from generate_dataset import generate
from optimize import optimize_support_map
from plate_models import PlateModel
from surrogate_model import UNet
from train import train

torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def test_direct_zero_and_loaded(self):
        mask = np.zeros((17, 17))
        model = PlateModel(mask, 2)
        zero = model.solve_fvk(0)
        np.testing.assert_allclose(zero['w'], 0, atol=1e-12)
        loaded = model.solve_fvk(-100, maxit=200)
        self.assertTrue(loaded['info']['converged'])
        self.assertLess(loaded['w'].min(), 0)
        np.testing.assert_allclose(loaded['w'], loaded['w'][::-1], atol=1e-8)

    def test_exciton_manufactured_and_positive(self):
        n = 17
        width = 2e-6
        x = np.linspace(0, width, n)
        xx, yy = np.meshgrid(x, x)
        expected = np.sin(np.pi * xx / width) * np.sin(np.pi * yy / width)
        d, tau = .97e-4, 1.02e-9
        eigenvalue = 8 * d / (width/(n-1))**2 * np.sin(np.pi/(2*(n-1)))**2
        result = solve_exciton_fixed(np.zeros((n,n)), width, expected*(eigenvalue+1/tau))
        np.testing.assert_allclose(result, expected, atol=1e-12)
        result = solve_exciton_fixed(.002*expected, width, gaussian_G(n, width))
        self.assertGreaterEqual(result.min(), 0)
        np.testing.assert_allclose(bernoulli(np.array([0., 1.]))[0], 1)

    def test_generate_train_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = generate(Path(tmp)/'toy.npz', n=10, grid=32, width_um=2, q=-100,
                            classes=['single_circle_center'])
            with np.load(path) as data:
                self.assertEqual(data['strain'].shape, (10,1,32,32))
                self.assertTrue(np.isfinite(data['strain']).all())
            report = train(path, Path(tmp)/'training', epochs=1, batch_size=2)
            self.assertTrue(np.isfinite(report['test_mse']))
            self.assertEqual(len(set(sum(report['splits'].values(), []))), 10)
            model = UNet()
            model.load_state_dict(torch.load(Path(tmp)/'training/best.pth', weights_only=True))
            self.assertTrue((Path(tmp)/'training/metrics.json').is_file())

    def test_optimization_constraints(self):
        class ToySurrogate(torch.nn.Module):
            def forward(self, x):
                return .01 + .02 * torch.nn.functional.avg_pool2d(x[:, :1], 3, 1, 1)
        support = torch.zeros(1,1,256,256)
        support[..., 120:136, :] = 1
        binary, soft, losses = optimize_support_map(ToySurrogate(), support, 128,128,12,12,2)
        self.assertTrue(np.isfinite(losses['total']).all())
        np.testing.assert_array_equal(binary, binary[::-1])
        np.testing.assert_array_equal(binary, binary[:,::-1])
        np.testing.assert_array_equal(binary[88:168], support.numpy()[0,0,88:168])
        self.assertTrue(np.isin(binary, [0,1]).all())
        self.assertTrue(np.isfinite(soft).all())


if __name__ == '__main__':
    unittest.main()
