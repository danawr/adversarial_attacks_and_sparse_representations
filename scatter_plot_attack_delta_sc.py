from absl import app, flags
import torch
import torch.nn.functional as F
import torchattacks
from models import SC_models
from models import wide_resnet
import utils
import tqdm
import io
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from attacks import square_attack
FLAGS = flags.FLAGS


def main(_):
    attack_type = FLAGS.attack_type
    sc_path = FLAGS.sc_path
    classifier_path = FLAGS.classifier_path
    batch_size = 32
    eps = 8.0 / 255
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data and models.
    data = utils.ld_cifar10(batch_size)
    net = wide_resnet.WideResNet(num_classes=10, depth=28, width=10,
                                 activation_fn=wide_resnet.Swish, mean=wide_resnet.CIFAR10_MEAN,
                                 std=wide_resnet.CIFAR10_STD)
    SC_net = SC_models.LISTAConvDict(num_input_channels=3, num_output_channels=3)

    with open(sc_path, 'rb') as f:
        buf = io.BytesIO(f.read())
        state_dict = torch.load(buf)
    SC_net.load_state_dict(state_dict)
    with open(classifier_path, 'rb') as f:
        buf = io.BytesIO(f.read())
        state_dict = torch.load(buf)
    net.load_state_dict(state_dict)

    if device == "cuda":
        net = net.cuda()
        SC_net = SC_net.cuda()

    for batch_data in tqdm.tqdm(data.test):
        x, y = batch_data
        x, y = x.squeeze().to(device), y.squeeze().to(device)

        pre_logits = net(x)
        _, y_pred = pre_logits.max(1)  # model prediction on clean examples
        if attack_type == 'PGD':
            x_attacked = projected_gradient_descent(net, x.detach(), eps, 0.01, 40, np.inf, y=y_pred.detach())
        else:  # AutoAttack.
            atk = torchattacks.AutoAttack(net, eps=eps)
            x_attacked = atk(x, y_pred)
        z_attacked = SC_net.forward_enc(x_attacked)
        pre_logits_pgd = net(x_attacked)
        _, y_pred_pgd = pre_logits_pgd.max(1)

        # Noise
        noise = torch.randn_like(x)
        noise_norms = noise.view(y.shape[0], -1).norm(p=float('inf'), dim=1)
        noise.div_(noise_norms.view(-1, 1, 1, 1))
        z_noise = SC_net.forward_enc(x + eps * noise)

        # Scatter the sparse code DIFFS against the original component norm.
        z = SC_net.forward_enc(x)  # clean sparse code
        z_attacked -= z
        z_noise -= z
        ind = torch.nonzero(y != y_pred_pgd)[0]
        plt.figure()
        plt.scatter(abs(z[ind, :, :, :].flatten().cpu().detach().numpy()),
                    abs(z_attacked[ind, :, :, :].flatten().cpu().detach().numpy()), label='AutoAttack', marker='+',
                    alpha=0.3,
                    color='#03719C')
        plt.scatter(abs(z[ind, :, :, :].flatten().cpu().detach().numpy()),
                    abs(z_noise[ind, :, :, :].flatten().cpu().detach().numpy()), label='noise', marker='o', alpha=0.2,
                    edgecolors='darkgray', color='#fd7f6f')
        plt.xlabel(r'|$\alpha_{Clean}$|')
        plt.ylabel(r'|$\Delta \alpha$|')
        plt.title('Classification: absolute difference in SC vs. magnitude.')
        plt.legend()
        plt.show()

        t_stat, p_value = ttest_ind(abs(z_attacked[ind, :, :, :].flatten().cpu().detach().numpy()),
                                    abs(z_noise[ind, :, :, :].flatten().cpu().detach().numpy()))
        print(f't_stat {t_stat}, p value {p_value}')
        print(f't_stat {t_stat}, p value {p_value}')
        break


if __name__ == "__main__":
    flags.DEFINE_string("attack_type", 'PGD', "The attack type to load (PGD / AutoAttack).")
    flags.DEFINE_string("sc_path", './checkpoints/sc_cifar10_1.pt', "Path to the trained sparse coder checkpoint")
    flags.DEFINE_string("classifier_path", './checkpoints/wrn_28_10_cifa10_best_itaration.pt', "Path to the trained classifier checkpoint")
    app.run(main)
