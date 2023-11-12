from absl import app, flags
import torch
from models import SC_models
import utils
import io
from matplotlib import pyplot as plt
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from sklearn.linear_model import Lasso
from scipy.stats import ttest_ind
FLAGS = flags.FLAGS


def main(_):
    sc_path = FLAGS.sc_path
    A_path = FLAGS.A_path
    m = FLAGS.m
    n = FLAGS.n
    K = FLAGS.K
    eps = 128.0 / 255

    net = SC_models.Lista(m, n, K)
    with open(sc_path, 'rb') as f:
        buf = io.BytesIO(f.read())
        state_dict = torch.load(buf)
    net.load_state_dict(state_dict)

    with open(A_path, 'rb') as f:
        buf = io.BytesIO(f.read())
        A = torch.load(buf)
    data = utils.ld_cs_synthetic_data(m, n, A=A)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()

    # Test
    net.eval()
    for batch_data in data.test:  # a single batch.
        x, y = batch_data
        x, y = x.squeeze().to(device), y.squeeze().to(device)
        x_hat = net(y)

        # PGD
        y_pgd = projected_gradient_descent(net, y.detach(), eps, 0.01, 40, 2, y=x_hat.detach())

        delta_pgd = y_pgd - y
        delta_pgd_norms = delta_pgd.view(delta_pgd.shape[0], -1).norm(p=2, dim=1)
        delta_pgd.div_(delta_pgd_norms.view(-1, 1))
        delta_pgd = eps * delta_pgd

        # Noise
        noise = torch.randn_like(y)
        noise_norms = noise.view(noise.shape[0], -1).norm(p=2, dim=1)
        noise.div_(noise_norms.view(-1, 1))
        noise = eps * noise


        lasso = Lasso(tol=0.001, alpha=0.00005, normalize=False)
        x_hat_pgd_diff = lasso.fit(A, np.transpose(delta_pgd.cpu().detach().numpy())).coef_
        pgd_mean_energy = utils.mean_energy(x_hat_pgd_diff)

        x_hat_only_noise = lasso.fit(A, np.transpose(noise.cpu().detach().numpy())).coef_
        noise_mean_energy = utils.mean_energy(x_hat_only_noise)

        pgd_diff_energy = np.nansum(x_hat_pgd_diff ** 2, axis=1)
        noise_energy = np.nansum(x_hat_only_noise ** 2, axis=1)
        t_stat, p_value = ttest_ind(pgd_diff_energy, noise_energy)
        print(f't_stat: {t_stat}, p value {p_value}')

        plt.figure()
        plt.rcParams['font.size'] = '16'
        plt.hist(np.nansum(x_hat_pgd_diff ** 2, axis=1), bins=30, color='#03719C', label='pgd', alpha=0.8)
        plt.hist(np.nansum(x_hat_only_noise ** 2, axis=1), bins=30, color='#fd7f6f', label='noise', alpha=0.8)
        plt.legend()
        plt.xlabel('Sparse code energy.')
        plt.title(f'Energy under the dictionary')
        plt.show()

        presentation_clip_value = 0.01
        x_hat_pgd_diff = np.where(abs(x_hat_pgd_diff) < presentation_clip_value, np.nan, x_hat_pgd_diff)
        x_hat_only_noise = np.where(abs(x_hat_only_noise) < presentation_clip_value, np.nan, x_hat_only_noise)
        plt.figure()
        plt.rcParams['font.size'] = '16'
        plt.hist(x_hat_pgd_diff.reshape(-1), bins=30, color='#03719C', label='pgd', alpha=0.8)
        plt.hist(x_hat_only_noise.reshape(-1), bins=30, color='#fd7f6f', label='noise', alpha=0.8)
        plt.legend()
        plt.xlabel('Sparse code component size.')
        plt.title(f'Active values.')
        plt.show()
        print(f'Active values. PGD MSE: {pgd_mean_energy:.3f}. Noise MSE: {noise_mean_energy:.3f}')


if __name__ == "__main__":
    flags.DEFINE_string("sc_path", '/home/dana/adversarial_robustness/checkpoints/sc_9_m250_n500_K16.pt', "The path to the trained sparse coding network.")
    flags.DEFINE_string("A_path", '/home/dana/adversarial_robustness/checkpoints/sc_9_m250_n500_K16_A.pt',
                        "The path to the data creating dictionary on which the LISTA models were trained on.")
    flags.DEFINE_integer('m', 250, 'Synthetic dictionary data dim.')
    flags.DEFINE_integer('n', 500, 'Synthetic dictionary number of atoms.')
    flags.DEFINE_integer('K', 16, 'LISTA unfolding parameter.')
    app.run(main)
