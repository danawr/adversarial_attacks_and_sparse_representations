from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import logging
import scipy
import scipy.signal
from itertools import islice


class FCBlock(torch.nn.Module):
    """One iteration of LISTA."""

    def __init__(self, first_dim, second_dim):
        super(FCBlock, self).__init__()
        self.model = nn.Sequential(nn.Linear(first_dim, second_dim),
                                   nn.ReLU())

    def forward(self, x):
        return self.model(x)


class FCNet(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self, measurement_dim, input_dim, hidden_dim=500, num_layers=16):
        super(FCNet, self).__init__()
        layers = []
        layers.append(FCBlock(measurement_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(FCBlock(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def ld_cifar10(batch_size=512, shuffle_train=True):
    """Load training and test data."""
    train_transforms = torchvision.transforms.Compose(
        [#torchvision.transforms.RandomCrop(32, padding=4),
         # torchvision.transforms.RandomHorizontalFlip(0.5),
         torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="/tmp/data", train=True, transform=train_transforms, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="/tmp/data", train=False, transform=test_transforms, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


def ld_mnist(batch_size=512):
    """Load training and test data."""
    train_transforms = torchvision.transforms.Compose(
        [#torchvision.transforms.RandomCrop(32, padding=4),
         torchvision.transforms.RandomHorizontalFlip(0.5),
         torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="/tmp/data", train=True, transform=train_transforms, download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="/tmp/data", train=False, transform=test_transforms, download=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


def ld_tiny_imagenet(batch_size):
    train_transforms = torchvision.transforms.Compose(
        [#torchvision.transforms.RandomCrop(32, padding=4),
         torchvision.transforms.RandomHorizontalFlip(0.5),
         torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_dataset = torchvision.datasets.ImageFolder('tiny-imagenet-200/train', transform=train_transforms)
    test_dataset = torchvision.datasets.ImageFolder('tiny-imagenet-200/val', transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


class SCDataset(torch.utils.data.Dataset):
    def __init__(self, m, n, p_Bernoulli=0.1, A=None):
        super(SCDataset).__init__()
        self.m = m
        self.n = n
        self.p_Bernoulli = p_Bernoulli
        self.normal_sampler = torch.distributions.Normal(0, 1.0)
        self.len = 200000
        if A is not None:
            self.A = A
        else:
            self.A = F.normalize(self.normal_sampler.sample([m, n]) / m, p=2, dim=0)

    def sample_x(self, num_samples=1):
        x = torch.zeros([self.n, num_samples]).bernoulli_(self.p_Bernoulli)
        x *= self.normal_sampler.sample(x.shape)
        return x.squeeze()

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        x = self.sample_x(num_samples=1)
        y = self.A @ x
        return x, y


class SCTestSet(SCDataset):
    def __init__(self, sc_dataset, test_set_size=1000):
        super().__init__(sc_dataset.m, sc_dataset.n, sc_dataset.p_Bernoulli, sc_dataset.A)
        self.test_set_size = test_set_size
        self.test_x = None

    def __len__(self):
        # return self.test_set_size
        return 1

    def __getitem__(self, item):
        if self.test_x is None:
            self.test_x = self.sample_x(num_samples=self.test_set_size).squeeze()
        y = self.A @ self.test_x
        return self.test_x.transpose(1, 0), y.transpose(1, 0)


def ld_cs_synthetic_data(m, n, batch_size=512, A=None):
    """Load training and test data."""

    train_dataset = SCDataset(m, n, A=A)
    test_dataset = SCTestSet(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1
    )
    return EasyDict(train=train_loader, test=test_loader)
    # return train_dataset, test_dataset


def get_logger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(path))
    return logger


def validate(net, data, eps, iterations, test=False, device='cuda'):
    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_pgd=0)
    if test:
        # Test on all test data but two validation batches.
        start, end = 0, len(data.dataset)
    else:
        # Validate on the first two batches.
        start, end = 0, 2
    for x, y in islice(data, start, end):
        x, y = x.to(device), y.to(device)
        # x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, 3, 1, 1)
        _, y_pred = net(x).max(1)  # model prediction on clean examples
        # x_pgd = projected_gradient_descent(net, x, eps, 0.01, iterations, np.inf, y=y_pred)
        # _, y_pred_pgd = net(x_pgd).max(1) # model prediction on PGD adversarial examples
        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        # report.correct_pgd += y_pred_pgd.eq(y).sum().item()
    return report.correct / report.nb_test, None #report.correct_pgd / report.nb_test


def get_spectral_reg(model):
    l_reg = 0.0
    # counter = 0
    for ind, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            w_mat = m.weight.reshape(m.weight.size(0), -1)
            # w_mat = reshape_weight_to_matrix(m.weight)
            svdvals = torch.linalg.svdvals(w_mat)
            l_reg += (svdvals.max() - svdvals.min())**2
            # l_reg += (svdvals.max() - 1) ** 2
            # l_reg += (svdvals.min() - 1) ** 2
            # sigma_max = torch.linalg.matrix_norm(w_mat, ord=2)
            # sigma_min = torch.linalg.matrix_norm(w_mat, ord=-2)
            # l_reg += sigma_max
            # # print(m.input_size)
            # w_fft = F.pad(m.weight, (0, m.input_size[0].int() - m.weight.size(2), 0, m.input_size[1].int() - m.weight.size(3)))
            # w_fft = torch.fft.rfft2(w_fft)
            # l_reg += torch.norm(w_fft, p=2, dim=-1).max()
    return l_reg


def print_spectral_extrema(model):
    for ind, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            w_mat = m.weight.reshape(m.weight.size(0), -1)
            svdvals = torch.linalg.svdvals(w_mat)
            # svdvals = torch.linalg.svdvals(w_mat)
            print(f'module {ind}: {svdvals.max()}, {svdvals.min()}, cond: {svdvals.max() / svdvals.min()}, non zeros:{sum(svdvals > 0)}/{len(svdvals)}')


def mean_energy(signal_batches):
    if isinstance(signal_batches, list):
        return [np.mean(np.nansum(signal_batch ** 2, axis=1)) for signal_batch in signal_batches]
    return np.mean(np.nansum(signal_batches ** 2, axis=1))
    # return np.nanmean(np.nanmax(abs(signal_batches), axis=1))


def get_excess_sup(x_list, x_orig):
    return [np.where((x_orig == 0) & (x != 0), x, np.nan) for x in x_list]


def get_in_sup(x_list, x_orig):
    return [np.where((x_orig != 0) & (x != 0), x - x_orig, np.nan) for x in x_list]


def get_missing_sup(x_list, x_orig):
    return [np.where((x_orig != 0) & (x == 0), x_orig, np.nan) for x in x_list]


def get_error_bar(x_list, x_orig):
    low_p = [np.percentile(np.nansum((x - x_orig) ** 2 , axis=1), 0.25) for x in x_list]
    high_p = [np.percentile(np.nansum((x - x_orig) ** 2, axis=1), 0.75) for x in x_list]
    return np.stack(zip(low_p, high_p)).T


def visualize_signals(x_orig, x_pgd, x_da, x_clean, algo_name):
    # Excess components error
    x_clean_excess_sup = np.where((x_orig == 0) & (x_clean != 0), x_clean, np.nan)
    x_da_excess_sup = np.where((x_orig == 0) & (x_da != 0), x_da, np.nan)
    x_pgd_excess_sup = np.where((x_orig == 0) & (x_pgd != 0), x_pgd, np.nan)

    # Error in the correct domain
    x_clean_in_sup = np.where((x_orig != 0) & (x_clean != 0), x_clean - x_orig, np.nan)
    x_da_in_sup = np.where((x_orig != 0) & (x_pgd != 0), x_da - x_orig, np.nan)
    x_pgd_in_sup = np.where((x_orig != 0) & (x_pgd != 0), x_pgd - x_orig, np.nan)

    # Missing components error
    x_clean_missing_sup = np.where((x_orig != 0) & (x_clean == 0), x_orig, np.nan)
    x_da_missing_sup = np.where((x_orig != 0) & (x_da == 0), x_orig, np.nan)
    x_pgd_missing_sup = np.where((x_orig != 0) & (x_pgd == 0), x_orig, np.nan)

    # Error energy decomposition
    x = ['DA', 'PGD', 'Clean']
    excess_stack = np.array([mean_energy(x) for x in [x_da_excess_sup, x_pgd_excess_sup, x_clean_excess_sup]])
    in_sup_stack = np.array([mean_energy(x_da_in_sup), mean_energy(x_pgd_in_sup), mean_energy(x_clean_in_sup)])
    missing_stack = np.array(
        [mean_energy(x_da_missing_sup), mean_energy(x_pgd_missing_sup), mean_energy(x_clean_missing_sup)])
    error_bars = get_error_bar([x_da, x_pgd, x_clean], x_orig)

    r = 0.45
    plt.figure()
    plt.rcParams['font.size'] = '16'
    plt.bar(x, excess_stack, r, edgecolor='white', color='#8bd3c7')
    plt.bar(x, in_sup_stack, r, bottom=excess_stack, edgecolor='white', color='#03719C')
    plt.bar(x, missing_stack, r, bottom=excess_stack + in_sup_stack, yerr=error_bars, edgecolor='white',
            color='#fd7f6f')
    plt.ylabel("Error energy")
    plt.legend(["Excess", "In support", "Missing"])
    plt.title(f"{algo_name}. Error energy source.")
    plt.show()


def visualize_sparsity_awareness(x_orig, x_list_1, x_list_2, algo_name):

    excess_sups_1 = get_excess_sup(x_list_1, x_orig)
    in_sups_1 = get_in_sup(x_list_1, x_orig)
    missing_sups_1 = get_missing_sup(x_list_1, x_orig)

    excess_sups_2 = get_excess_sup(x_list_2, x_orig)
    in_sups_2 = get_in_sup(x_list_2, x_orig)
    missing_sups_2 = get_missing_sup(x_list_2, x_orig)

    # Error energy decomposition
    x = ['DA', 'PGD', 'Clean']
    x_axis = np.arange(len(x))
    excess_stack_1 = np.array(mean_energy(excess_sups_1))
    in_sup_stack_1 = np.array(mean_energy(in_sups_1))
    missing_stack_1 = np.array(mean_energy(missing_sups_1))

    excess_stack_2 = np.array(mean_energy(excess_sups_2))
    in_sup_stack_2 = np.array(mean_energy(in_sups_2))
    missing_stack_2 = np.array(mean_energy(missing_sups_2))

    r = 0.45
    plt.figure()
    plt.rcParams['font.size'] = '16'
    plt.bar(x_axis - r/2, excess_stack_1, r, edgecolor='white', color='#8bd3c7')
    plt.bar(x_axis - r/2, in_sup_stack_1, r, bottom=excess_stack_1, edgecolor='white', color='#03719C')
    plt.bar(x_axis - r / 2, missing_stack_1, r, bottom=excess_stack_1 + in_sup_stack_1,
            yerr=get_error_bar(x_list_1, x_orig), edgecolor='white', color='#fd7f6f')

    plt.bar(x_axis + r/2, excess_stack_2, r, edgecolor='white', color='#8bd3c7')
    plt.bar(x_axis + r/2, in_sup_stack_2, r, bottom=excess_stack_2, edgecolor='white', color='#03719C')
    plt.bar(x_axis + r / 2, missing_stack_2, r, bottom=excess_stack_2 + in_sup_stack_2,
            yerr=get_error_bar(x_list_2, x_orig), edgecolor='white', color='#fd7f6f')

    plt.xticks(x_axis, x)
    plt.ylabel("Error energy")
    plt.legend(["Excess", "In support", "Missing"])
    plt.title(f"{algo_name}. Error energy source.")
    plt.show()


def block_toeplitz(c, r=None):
    '''
    Construct a block Toeplitz matrix, with blocks having the same shape

    Signature is compatible with ``scipy.linalg.toeplitz``

    Parameters
    ----------
    c : list of 2d arrays
        First column of the matrix.
        Each item of the list should have same shape (mb,nb)
    r : list of 2d arrays
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        made of blocks ``[c[0], r[1:]]``.

    c and r can also be lists of scalars; if so they will be broadcasted
    to the fill the blocks

    Returns
    -------
    A : (len(c)*mb, len(r)*nb) ndarray
        The block Toeplitz matrix.
    '''
    c = [np.atleast_2d(ci) for ci in c]
    if r is None:
        r = [np.conj(ci) for ci in c]
    else:
        r = [np.atleast_2d(rj) for rj in r]

    mb,nb = c[0].shape
    dtype = (c[0]+r[0]).dtype
    m = len(c)
    n = len(r)

    A = np.zeros((m*mb, n*nb), dtype=dtype)

    for i in range(m):
        for j in range(n):
            # 1. select the Aij block from c or r:
            d = i-j
            if d>=0:
                Aij = c[d]
            else:
                Aij = r[-d]
            # 2. paste the block
            A[i*mb:(i+1)*mb, j*mb:(j+1)*mb] = Aij

    return A


def col_Toeplitz_block(kernel_col, input_h=32):
    # Assumes the kernel is a square.
    d = len(kernel_col)
    row = np.zeros(input_h)
    row[:(d // 2) + 1] = kernel_col[(d // 2):]
    col = np.zeros(input_h)
    col[:(d // 2) + 1] = np.flip(kernel_col[:(d // 2) + 1])
    return scipy.linalg.toeplitz(col, row)


def convert_conv_dict_to_fc(conv_dict, input_shape):
    # assumes zero padding for 'same' convolution.
    input_c_k, n_k, k_h, k_w = conv_dict.shape
    _, input_c, h, w = input_shape
    conv_dict = conv_dict.cpu().detach().numpy()
    all_blocks = []
    for c in range(input_c):
        channel_blocks = []
        for k in range(n_k):
            kernel = conv_dict[c, k, :, :].T
            zeros_mat = np.zeros((h, h))
            col_blocks = list(np.zeros(w))
            col_blocks[:(k_w//2 + 1)] = [col_Toeplitz_block(kernel[:, j], input_h=h) for j in np.arange(k_w//2, -1, -1)]
            col_blocks[(k_w // 2 + 1):] = [zeros_mat for _ in np.arange(k_w // 2 + 1, w)]
            row_blocks = list(np.zeros(w))
            row_blocks[:k_w // 2 + 1] = [col_Toeplitz_block(kernel[:, j], input_h=h) for j in np.arange(k_w // 2, k_w)]
            row_blocks[k_w // 2 + 1:] = [zeros_mat for _ in np.arange(k_w // 2 + 1, w)]
            channel_blocks.append(block_toeplitz(col_blocks, row_blocks))
        all_blocks.append(np.concatenate(channel_blocks, axis=1))
    return np.concatenate(all_blocks, axis=0)



