from absl import app, flags
from attacks import dict_attack
from art.attacks.evasion import UniversalPerturbation, TargetedUniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10, load_mnist
from collections import defaultdict
from models import SC_models
from models import wide_resnet
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision
import utils
import tqdm
import io
import numpy as np

FLAGS = flags.FLAGS

def main(_):

    # Compare DA to baseline attacks on real data and networks.
    # SC_net is a classifier that is only used to pick the correct DA / targeted attack.

    dataset = FLAGS.dataset
    classifier_path = FLAGS.classifier_path
    sc_classifier_path = FLAGS.sc_classifier_path
    attacks_dir = FLAGS.attacks_dir
    calculate_attacks = FLAGS.calculate_attacks
    batch_size = 32
    eps = 5  # Just for the attacks calculations, not used at inference.

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset == 'MNIST':
        data = utils.ld_mnist(batch_size)
        net = torchvision.models.resnet18(num_classes=10)
        network_name = 'ResNet18'
    elif dataset == 'CIFAR10':
        data = utils.ld_cifar10(batch_size)
        net = wide_resnet.WideResNet(num_classes=10, depth=28, width=10,
                                     activation_fn=wide_resnet.Swish, mean=wide_resnet.CIFAR10_MEAN,
                                     std=wide_resnet.CIFAR10_STD)
        network_name = 'WideResNet28'
    else:
        print(
            'Only MNIST and CIFAR10 are supported. '
            'For other datasets, please train proper regular and sparse code classifiers.')


    # Task network ("real" classifier).
    with open(classifier_path, 'rb') as f:
        buf = io.BytesIO(f.read())
        state_dict = torch.load(buf)
    net.load_state_dict(state_dict)

    # Sparse coding classifier.
    input_channels = 1 if dataset == 'MNIST' else 3
    input_size = 28 if dataset == 'MNIST' else 32
    dummy_SC_net = SC_models.LISTAConvDict(num_input_channels=input_channels, num_output_channels=input_channels)
    SC_net = SC_models.SCClassifier(sparse_coder=dummy_SC_net, code_dim=(input_size ** 2) * 64)
    del dummy_SC_net
    with open(sc_classifier_path, 'rb') as f:
        buf = io.BytesIO(f.read())
        state_dict = torch.load(buf)
    SC_net.load_state_dict(state_dict)

    if calculate_attacks:
        # Calculate baseline universal adversarial attack.
        if dataset == 'MNIST':
            (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
        else:
            (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        criterion = nn.CrossEntropyLoss()
        classifier = PyTorchClassifier(
            model=SC_net,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=criterion,
            input_shape=(input_channels, input_size, input_size),
            nb_classes=10,
        )
        attack = UniversalPerturbation(classifier, attacker='pgd', eps=eps, norm=2, delta=0.5, max_iter=5,
                                       verbose=False)
        _ = attack.generate(x_train[:2000], y_train[:2000])
        np.save(f'./{dataset.lower()}_uap', attack.noise)
        x_universal_attack = torch.tensor(attack.noise).cuda()

        # Calculate targeted UAP (to all of the target classes, and apply like we apply DA by the runner up).
        num_examples_for_uap = 10000
        targeted_uaps = {}
        targeted_attack = TargetedUniversalPerturbation(classifier, attacker='fgsm', attacker_params={'targeted': True},
                                                        delta=0.2, max_iter=20, eps=eps, norm=2)
        for target_class in range(10):
            targeted_labels = np.zeros_like(y_train[:num_examples_for_uap])
            targeted_labels[:, target_class] = 1
            _ = targeted_attack.generate(x_train[:num_examples_for_uap], targeted_labels)
            targeted_uaps[target_class] = targeted_attack.noise
        np.save(f'{attacks_dir}/{dataset.lower()}_targetes_uap_dict', targeted_uaps)

        # Calculate DA
        linear_classifier = SC_net.classifier.weight.data.detach().cpu().numpy()
        conv_dict = SC_net.conv_dictionary().data
        A = utils.convert_conv_dict_to_fc(conv_dict, input_shape=(None, input_channels, input_size, input_size))
        A_col_norms = np.linalg.norm(A, axis=0)
        print('Started SVD.')
        u, s, vh = np.linalg.svd(A, full_matrices=False)
        print('Finished SVD.')
        mult = A_col_norms * vh
        cons_factor = vh @ mult.T

        # Calculate the classification (source - target) DA.
        attacks_dict = defaultdict()
        for c_src in range(10):
            for c_tgt in range(c_src + 1, 10):
                print(f'{c_src}_{c_tgt}')
                x_da_vec, _ = dict_attack.get_classifier_attack_delta(s, vh, A, cons_factor, eps,
                                                                      linear_classifier, c_src, c_tgt)
                attacks_dict[f'{c_src}_{c_tgt}'] = x_da_vec
        np.save(f'{attacks_dir}/{dataset.lower()}_da_dict', attacks_dict)
    else:
        x_universal_attack = np.load(f'{attacks_dir}/{dataset.lower()}_uap.npy', allow_pickle=True)
        x_universal_attack = torch.tensor(x_universal_attack).cuda()
        targeted_uaps = np.load(f'{attacks_dir}/{dataset.lower()}_targetes_uap_dict.npy', allow_pickle=True).item()
        attacks_dict = np.load(f'{attacks_dir}/{dataset.lower()}_da_dict.npy', allow_pickle=True).item()


    if device == "cuda":
        net = net.cuda()
        SC_net = SC_net.cuda()


    # epsilon_values = np.linspace(0, 1, 10)
    epsilon_values = np.linspace(1, 10, 5)
    DA_acc = []
    UDA_acc = []
    UDA_acc_targeted = []
    noise_acc = []
    for eps in epsilon_values:
        report = utils.EasyDict(nb_test=0, correct=0, correct_pgd=0, correct_da=0, correct_uda=0,
                                correct_uda_targeted=0, correct_noise=0, da_sc_error=0, clean_sc_error=0,
                                noise_sc_error=0, pgd_sc_error=0)

        for batch_data in tqdm.tqdm(data.test):
            x, y = batch_data
            x, y = x.squeeze().to(device), y.squeeze().to(device)
            if dataset == 'MNIST':
                x = torch.unsqueeze(x, dim=1)

            pre_logits_sc = SC_net(x)
            _, y_pred_sc = pre_logits_sc.max(1)
            c_tgt = torch.argsort(pre_logits_sc, dim=1, descending=True)[:, 1]
            c_tgt = torch.where(c_tgt == y, torch.argsort(pre_logits_sc, dim=1, descending=True)[:, 0], c_tgt)

            # Clean
            if dataset == 'MNIST':
                x = x.repeat(1, 3, 1, 1)
            pre_logits = net(x)
            _, y_pred = pre_logits.max(1)

            # UDA
            x_universal_attack = (eps / torch.norm(x_universal_attack)) * x_universal_attack
            x_uda = torch.tile(x_universal_attack, (x.shape[0], 1, 1, 1))
            pre_logits_uda = net(x + x_uda)
            _, y_pred_uda = pre_logits_uda.max(1)

            # Targeted UDA
            x_uda_targeted = []
            for target_class in c_tgt.detach().cpu().numpy():
                attack = torch.squeeze(torch.Tensor(targeted_uaps[target_class])).cuda()
                orig_norm = torch.norm(attack, p=2)
                x_uda_targeted.append(eps * attack / orig_norm)
            x_uda_targeted = torch.stack(x_uda_targeted)
            if dataset == 'MNIST':
                x_uda_targeted = torch.unsqueeze(x_uda_targeted, dim=1)
                x_uda_targeted = x_uda_targeted.repeat(1, 3, 1, 1)
            pre_logits_uda_targeted = net(x + x_uda_targeted)
            _, y_pred_uda_targeted = pre_logits_uda_targeted.max(1)


            # DA
            # Per source - target pair
            x_da = []
            key_1 = torch.where(y < c_tgt, y, c_tgt)
            key_2 = torch.where(y < c_tgt, c_tgt, y)
            sign = torch.where(y < c_tgt, 1, -1)
            for i in range(x.shape[0]):
                key = f'{key_1[i]}_{key_2[i]}'
                attack = torch.Tensor(attacks_dict[key]).reshape(input_channels, input_size, input_size).cuda()
                orig_norm = torch.norm(attack, p=2)
                attack = eps * attack / orig_norm
                x_da.append(sign[i] * attack)
            x_da = torch.stack(x_da)
            if dataset == 'MNIST':
                x_da = x_da.repeat(1, 3, 1, 1)
            pre_logits_da = net(x + x_da)
            _, y_pred_da = pre_logits_da.max(1)

            # Noise
            noise = torch.randn_like(x[0])
            noise = (eps / torch.norm(noise)) * noise
            noise = torch.tile(noise, (x.shape[0], 1, 1, 1))
            pre_logits_noise = net(x + noise)
            _, y_pred_noise = pre_logits_noise.max(1)

            # Log
            report.nb_test += y.size(0)
            report.correct += y_pred.eq(y).sum().item()
            report.correct_da += y_pred_da.eq(y).sum().item()
            report.correct_uda += y_pred_uda.eq(y).sum().item()
            report.correct_uda_targeted += y_pred_uda_targeted.eq(y).sum().item()
            report.correct_noise += y_pred_noise.eq(y).sum().item()


        print(f"Best iteration: test acc on clean examples: {report.correct / report.nb_test:.3f}")
        print(f"Best iteration: test acc on noisy examples: {report.correct_noise / report.nb_test:.3f}")
        print(f"Best iteration: test acc on UAP adversarial examples: {report.correct_uda / report.nb_test:.3f}")
        print(f"Best iteration: test acc on targeted UAP adversarial examples: {report.correct_uda_targeted / report.nb_test:.3f}")
        print(f"Best iteration: test acc on DA adversarial examples: {report.correct_da / report.nb_test:.3f}")

        DA_acc.append(report.correct_da / report.nb_test)
        UDA_acc.append(report.correct_uda / report.nb_test)
        UDA_acc_targeted.append(report.correct_uda_targeted / report.nb_test)
        noise_acc.append(report.correct_noise / report.nb_test)

    # np.save(f'./{dataset.lower()}_da_accuracy', DA_acc)
    # np.save(f'./{dataset.lower()}_uda_accuracy', UDA_acc)
    # np.save(f'./{dataset.lower()}_targeted_uda_accuracy', UDA_acc_targeted)
    # np.save(f'./{dataset.lower()}_noise_accuracy', noise_acc)
    

    plt.figure()
    plt.plot(epsilon_values, DA_acc, label='DA', color='#03719C')
    plt.plot(epsilon_values, UDA_acc, label='UAP', color='#8bd3c7', linestyle='dashed')
    plt.plot(epsilon_values, UDA_acc_targeted, label='Targeted UAP', color='#8bd3c7')
    plt.plot(epsilon_values, noise_acc, label='noise', color='#fd7f6f')
    plt.xlim(epsilon_values[0], epsilon_values[-1])
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{network_name} accuracy on {dataset} vs. attack budget.')
    plt.show()

if __name__ == "__main__":
    flags.DEFINE_string("dataset", 'MNIST', "The classification dataset (MNIST / CIFAR10).")
    flags.DEFINE_string("attacks_dir", './attack_accuracy', "The directory to save/ load calculated attacks.")
    flags.DEFINE_string("classifier_path", './checkpoints/resnet18_mnist_best_iteration.pt', "Path to the trained classifier checkpoint")
    flags.DEFINE_string("sc_classifier_path", './checkpoints/classify_mnist_learn_last_layer.pt', "Path to the trained sparse coding linear classifier checkpoint")
    flags.DEFINE_bool(
        "calculate_attacks", False, "Whether to calculate (True) or to load (False) the adversarial attacks."
    )

    app.run(main)