from absl import app, flags
import torch
from models import SC_models
import utils
import io
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import levene
FLAGS = flags.FLAGS


def main(_):
    saved_corr_dir = FLAGS.saved_corr_dir
    dataset = FLAGS.dataset

    if dataset == 'TinyImageNet':
        pgd_corr_sums_data = np.load(
            f'{saved_corr_dir}/pgd_corr_sums_{dataset}_l2_data_dict_PGD.npy')
        pgd_corr_sums_random = []
        for part in range(16):
            pgd_corr_sums_random.append(np.load(f'{saved_corr_dir}/pgd_corr_sums_{dataset}_l2_full_random_{part}_dict_PGD.npy'))
        pgd_corr_sums_random = np.concatenate(pgd_corr_sums_random)
        noise_corr_sums_data = np.load(
            f'{saved_corr_dir}/noise_corr_sums_{dataset}_l2_data_dict_PGD.npy')
        pgd_atom_probability_data = pgd_corr_sums_data / np.sum(pgd_corr_sums_data)
        pgd_atom_probability_random = pgd_corr_sums_random / np.sum(pgd_corr_sums_random)
        noise_atom_probability_data = noise_corr_sums_data / np.sum(noise_corr_sums_data)

        # Plot correlations histograms
        min_value = np.minimum(np.min(pgd_atom_probability_data), np.min(pgd_atom_probability_random))
        max_value = np.maximum(np.max(pgd_atom_probability_data), np.max(pgd_atom_probability_random))
        bins = np.linspace(min_value, max_value, 50)
        plt.figure()
        plt.hist(pgd_atom_probability_data, bins=bins, color='#03719C', label=f'PGD data dict', alpha=0.5)
        plt.hist(pgd_atom_probability_random, bins=bins, color='#fd7f6f', label=f'PGD random dict', alpha=0.5)
        plt.hist(noise_atom_probability_data, bins=bins, color='#8bd3c7', label='noise', alpha=0.5)
        plt.legend()
        plt.xlabel('Normalized sum of component correlations')
        plt.title(f'Atom "probability" distribution. {dataset}, l2.')
        plt.show()

    else:  # CIFAR10
        dict_mode = FLAGS.dict_mode
        if dict_mode == 'random':
            pgd_corrs = np.load(f'{saved_corr_dir}/pgd_corrs_l2_random_dict.npy',
                                    allow_pickle=True) / 10000
            noise_corrs = np.load(f'{saved_corr_dir}/noise_corrs_l2_random_dict.npy',
                                      allow_pickle=True) / 10000
            A = np.load(f'{saved_corr_dir}/A_l2_random_dict.npy', allow_pickle=True)
        else:
            pgd_corrs = np.load(f'{saved_corr_dir}/pgd_corr_sums_l2.npy', allow_pickle=True) / 10000
            noise_corrs = np.load(f'{saved_corr_dir}/noise_corr_sums_l2.npy', allow_pickle=True) / 10000

            sc_path = FLAGS.sc_path
            SC_net = SC_models.LISTAConvDict(num_input_channels=3, num_output_channels=3)
            with open(sc_path, 'rb') as f:
                buf = io.BytesIO(f.read())
                state_dict = torch.load(buf)
            SC_net.load_state_dict(state_dict)
            A = utils.convert_conv_dict_to_fc(SC_net.conv_dictionary, input_shape=(None, 3, 32, 32))
            del SC_net

        stat, p = levene(pgd_corrs, noise_corrs)
        print(f'{dataset}, {dict_mode}: stat {stat} p value {p}.')

        # Plot correlations histograms
        min_value = np.minimum(np.min(pgd_corrs / np.sum(pgd_corrs)), np.min(noise_corrs / np.sum(noise_corrs)))
        max_value = np.maximum(np.max(pgd_corrs / np.sum(pgd_corrs)), np.max(noise_corrs / np.sum(noise_corrs)))
        bins = np.linspace(min_value, max_value, 50)
        plt.figure()
        plt.hist(pgd_corrs / np.sum(pgd_corrs), bins=bins, color='#03719C', label='pgd', alpha=0.5)
        plt.hist(noise_corrs / np.sum(noise_corrs), bins=bins, color='#fd7f6f', label='noise', alpha=0.5)
        plt.legend()
        plt.xlabel('Normalized sum of component correlations')
        plt.title('Perturbation - Dictionary Atoms Correlation distribution.')
        plt.show()

        # Plot sub matrices spectra.
        sorted_inds_pgd = np.argsort(pgd_corrs)
        S = 30  # Number of atoms.
        A_top = A[:, sorted_inds_pgd[-2 * S:]]
        A_bottom = A[:, sorted_inds_pgd[:2 * S]]
        fig, ax = plt.subplots()
        custom_lines = [Line2D([0], [0], color='#03719C', lw=4),
                        Line2D([0], [0], color='#fd7f6f', lw=4),
                        Line2D([0], [0], color='#8bd3c7', lw=4)]
        for _ in range(50):
            top_submatrix = A_top[:, np.random.randint(0, A_top.shape[1], S)]
            s_top = np.linalg.svd(top_submatrix, full_matrices=False, compute_uv=False)
            ax.plot(np.arange(s_top.shape[0]), s_top, color='#03719C', alpha=0.5)

            bottom_submatrix = A_bottom[:, np.random.randint(0, A_bottom.shape[1], S)]
            s_bottom = np.linalg.svd(bottom_submatrix, full_matrices=False, compute_uv=False)
            ax.plot(np.arange(s_bottom.shape[0]), s_bottom, color='#fd7f6f', alpha=0.5)

            random_submatrix = A[:, np.random.randint(0, A.shape[1], S)]
            s_random = np.linalg.svd(random_submatrix, full_matrices=False, compute_uv=False)
            ax.plot(np.arange(s_random.shape[0]), s_random, color='#8bd3c7', alpha=0.5)
        ax.set_title(f'Singular Values, S = {S}')
        ax.legend(custom_lines, ['Top', 'Bottom', 'Random'])
        plt.show()


if __name__ == '__main__':
    flags.DEFINE_string('saved_corr_dir', './correlations', 'The directory in which the calculated correlations are saved.')
    flags.DEFINE_string('dataset', 'CIFAR10', 'The classification dataset (TinyImageNet / CIFAR10).')
    flags.DEFINE_string('dict_mode', 'data', 'Whether to plot for the trained or random dictionary (data/ random)')
    flags.DEFINE_string('sc_path', './checkpoints/sc_cifar10_1.pt', 'Path to the trained sparse coding linear classifier checkpoint')
    app.run(main)
