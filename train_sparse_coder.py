from absl import app, flags
import common
import torch
from torch.utils.tensorboard import SummaryWriter
import utils
from models import SC_models
import tqdm
FLAGS = flags.FLAGS


def main(_):
    dataset = FLAGS.dataset
    if dataset == 'synthetic':
        m = FLAGS.m
        n = FLAGS.n
        K = FLAGS.K
        experiment_name_suffix = f'_m{m}_n{n}_K{K}'
    else:
        experiment_name_suffix = ''
    experiment_name = f'{FLAGS.experiment_name_prefix}_{dataset}{experiment_name_suffix}'
    batch_size = FLAGS.batch_size
    lr = FLAGS.lr
    epochs = FLAGS.epochs
    model_name = FLAGS.model_name
    base_dir = FLAGS.base_dir

    logger = utils.get_logger(f'{base_dir}logs/{experiment_name}_log.txt')
    writer = SummaryWriter(f'{base_dir}/tensorboard/')
    # Log config
    logger.info(f'experiment_name: {experiment_name}, '
                f'dataset: {dataset}, '
                f'model name: {model_name}, '
                f'batch_size: {batch_size}, '
                f'initial lr: {lr}')

    # Load training and test data
    if dataset == 'synthetic':
        A = utils.get_sparse_dictionary(m, n)
        torch.save(A, f'{base_dir}checkpoints/{experiment_name}_A.pt')
        data = utils.ld_cs_synthetic_data(m, n, A=A)
    elif dataset == 'CIFAR10':
        data = utils.ld_cifar10(batch_size)
    elif dataset == 'MNIST':
        data = utils.ld_mnist(batch_size)
    elif dataset == 'TinyImageNet':
        data = utils.ld_tiny_imagenet(batch_size)
    else:
        print(
            'Only synthetic, CIFAR10, MNIST, and TinyImageNet are supported as dataset. '
            'For other options, please add a proper data loader.')
    num_channels = 1 if dataset == 'MNIST' else 3

    # Instantiate model, loss, and optimizer for training
    if dataset == 'synthetic':
        net = SC_models.Lista(m, n, K)
    else:
        net = SC_models.LISTAConvDict(num_input_channels=num_channels, num_output_channels=num_channels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    if dataset == 'synthetic':
        loss_function = torch.nn.MSELoss(reduction='mean')
    else:
        loss_function = common.get_criterion(
            losses_types=['msssim', 'l1'],
            factors=[0.8, 0.2],
            use_cuda=True
        )
    for epoch in range(epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}:")
        net.train()
        train_loss = 0.0
        for batch_num, batch_data in enumerate(tqdm.tqdm(data.train)):
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            if dataset == 'MNIST':
                x = torch.unsqueeze(x, dim=1)
            x_hat = net(y)
            if dataset is not 'synthetic':
                x_hat = x_hat[0]
            loss = loss_function(x, x_hat)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        logger.info(f"Train loss: {train_loss / batch_num:.4f}")

        # Test
        net.eval()
        for batch_data in data.test:
            x, y = batch_data
            x, y = x.squeeze().to(device), y.squeeze().to(device)
            if dataset == 'MNIST':
                x = torch.unsqueeze(x, dim=1)
            x_hat = net(y)
            if dataset is not 'synthetic':
                x_hat = x_hat[0]
            test_loss = loss_function(x, x_hat).item()
            test_error = torch.mean(torch.sum((x - x_hat)**2, dim=1) / torch.norm(x, dim=1)**2)
        torch.save(net.state_dict(), f'{base_dir}checkpoints/{experiment_name}.pt')
        writer.add_scalars('losses', {'train_loss': train_loss / batch_num, 'test_loss': test_loss}, epoch)
        writer.add_scalar('test_error', test_error, epoch)
    writer.close()


if __name__ == "__main__":
    flags.DEFINE_string("dataset", 'synthetic', "synthetic, CIFAR10, MNIST, and TinyImageNet are supported as dataset.")
    flags.DEFINE_string("model_name", 'lista', "Training model name.")
    flags.DEFINE_integer("m", 250, "Synthetic data dim.")
    flags.DEFINE_integer("n", 500, "Synthetic dictionary number of atoms.")
    flags.DEFINE_integer("K", 16, "LISTA unfolding parameter.")
    flags.DEFINE_integer("batch_size", 512, "Training batch_size.")
    flags.DEFINE_integer("l_r", 0.001, "Initial learning rate.")
    flags.DEFINE_integer("epochs", 1000, "Number of training epochs.")
    flags.DEFINE_string("experiment_name_prefix", 'sc', "Prefix of the saved model and data dictionary.")
    flags.DEFINE_string("base_dir", './', "The directory in which to save the trained model and data dictionary.")
    app.run(main)
