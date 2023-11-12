from absl import app, flags
from attacks import dict_attack
from models import SC_models
import torch
import utils
import io
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lasso

FLAGS = flags.FLAGS


def load_lista_model(path, m, n, K):
    net = SC_models.Lista(m, n, K)
    with open(path, 'rb') as f:
        buf = io.BytesIO(f.read())
        state_dict = torch.load(buf)
    net.load_state_dict(state_dict)
    return net


def npfy(var_list):
    return [var.cpu().detach().numpy() for var in var_list]


def main(_):
    lista_1_path = FLAGS.lista_1_path
    lista_2_path = FLAGS.lista_2_path
    A_path = FLAGS.A_path
    m = FLAGS.m
    n = FLAGS.n
    K = FLAGS.K
    eps = 128.0 / 255

    # Load models and data.
    net = load_lista_model(lista_1_path, m, n, K)
    # Load a second network to show transferability.
    net_2 = load_lista_model(lista_2_path, m, n, K)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        net = net.cuda()
        net_2 = net_2.cuda()

    with open(A_path, 'rb') as f:
        buf = io.BytesIO(f.read())
        A = torch.load(buf)
    data = utils.ld_cs_synthetic_data(m, n, A=A)

    loss_function = torch.nn.MSELoss(reduction='mean')
    for batch_data in data.test:
        x, y = batch_data
        x, y = x.squeeze().to(device), y.squeeze().to(device)
        x_clean = net(y)
        x_clean_2 = net_2(y)

        # PGD
        y_pgd = projected_gradient_descent(net, y.detach(), eps, 0.01, 40, 2, y=x_clean.detach())
        x_pgd = net(y_pgd)
        x_pgd_2 = net_2(y_pgd)

        # DA
        y_da = torch.Tensor(dict_attack.get_attack_delta_closed_form(A.cpu().detach().numpy(), eps)).to(device)
        y_da = torch.tile(y_da, (y.shape[0], 1))
        x_da = net(y + y_da)
        x_da_2 = net_2(y + y_da)

        clean_loss = loss_function(x, x_clean).item()
        x_orig, x_clean, x_pgd, x_da, x_clean_2, x_pgd_2, x_da_2, y, A = npfy(
            [x, x_clean, x_pgd, x_da, x_clean_2, x_pgd_2, x_da_2, y, A])

        utils.visualize_sparsity_awareness(x_orig, [x_da, x_pgd, x_clean], [x_da_2, x_pgd_2, x_clean_2],
                                           algo_name='Lista')

        # Compute classical sparse coding algorithms.
        def sparse_code_and_visualize(algo_name, tolerance):
            if algo_name == 'OMP':
                algo = OrthogonalMatchingPursuit(tol=tolerance, normalize=False)
            elif algo_name == 'Lasso':
                algo = Lasso(tol=tolerance, alpha=0.00005, normalize=False)
            else:
                print('Only Lasso and OMP are supported.')
            x_pgd_hat, x_da_hat, x_clean_hat = [algo.fit(A, np.transpose(signal)).coef_ for signal in
                                                [y_pgd.cpu().detach().numpy(), y + y_da.cpu().detach().numpy(), y]]
            utils.visualize_signals(x_orig, x_pgd_hat, x_da_hat, x_clean_hat, algo_name=algo_name)

        sparse_code_and_visualize('OMP', clean_loss)
        sparse_code_and_visualize('Lasso', clean_loss)


if __name__ == '__main__':
    flags.DEFINE_string('lista_1_path', './checkpoints/sc_9_m250_n500_K16.pt', 'The path to the first trained LISTA network.')
    flags.DEFINE_string('lista_2_path', './checkpoints/sc_9_m250_n500_K16_2.pt', 'The path to the second trained LISTA network.')
    flags.DEFINE_string('A_path', './checkpoints/sc_9_m250_n500_K16_A.pt',
                        'The path to the data creating dictionary on which the LISTA models were trained on.')
    flags.DEFINE_integer('m', 250, 'Synthetic dictionary data dim.')
    flags.DEFINE_integer('n', 500, 'Synthetic dictionary number of atoms.')
    flags.DEFINE_integer('K', 16, 'LISTA unfolding parameter.')
    app.run(main)
