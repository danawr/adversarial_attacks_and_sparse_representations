from absl import app, flags
import torch
from torch.utils.tensorboard import SummaryWriter
from models import SC_models
import utils
import tqdm
import io
import torch.nn.functional as F
FLAGS = flags.FLAGS


def main(_):
    # Trains a linear classifier over a learned sparse code (produced via a trained ConvLISTA).
    dataset = FLAGS.dataset
    experiment_name = FLAGS.experiment_name
    sc_path = FLAGS.sc_path
    batch_size = FLAGS.batch_size
    lr = FLAGS.lr
    epochs = FLAGS.epochs
    base_dir = FLAGS.base_dir
    model_name = 'conv_lista'
    logger = utils.get_logger(f'{base_dir}logs/{experiment_name}_log.txt')
    writer = SummaryWriter(f'{base_dir}/tensorboard/')
    # Log config
    logger.info(f'experiment_name: {experiment_name}, '
                f'linear classifier over sparse coding model: {model_name}, '
                f'trained sparse coder path: {sc_path}'
                f'batch_size: {batch_size}, '
                f'initial lr: {lr}.')

    # Load training and test data
    if dataset == 'CIFAR10':
        data = utils.ld_cifar10(batch_size)
    elif dataset == 'MNIST':
        data = utils.ld_mnist(batch_size)
    num_channels = 1 if dataset == 'MNIST' else 3
    input_size = 28 if dataset == 'MNIST' else 32

    # Instantiate model, loss, and optimizer for training
    SC_net = SC_models.LISTAConvDict(num_input_channels=num_channels, num_output_channels=num_channels)
    with open(sc_path, 'rb') as f:
        buf = io.BytesIO(f.read())
        state_dict = torch.load(buf)
    SC_net.load_state_dict(state_dict)
    net = SC_models.SCClassifier(sparse_coder=SC_net, code_dim=64 * input_size**2)
    del SC_net

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()

    # Train only the linear final layer!
    optimizer = torch.optim.Adam(net.classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     gamma=0.1,
                                                     milestones=[epochs // 2])
    loss_function_task = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch}/{FLAGS.nb_epochs}:")
        net.train()
        train_loss_task = 0.0
        train_loss_sc = 0.0
        train_loss = 0.0
        correct = 0.0
        nb_test = 0.0
        for num_batch, batch_data in enumerate(tqdm.tqdm(data.train)):
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            if dataset == 'MNIST':
                x = torch.unsqueeze(x, dim=1)
            logits = F.log_softmax(net(x), dim=1)
            loss = loss_function_task(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            _, y_pred = logits.max(1)
            correct += y_pred.eq(y).sum().item()
            nb_test += y.size(0)
        scheduler.step()
        train_loss = train_loss / num_batch
        train_acc = correct / nb_test * 100
        logger.info(f"Train loss: {train_loss:.4f}, train accuracy: {train_acc}")

        # Test
        net.eval()
        test_loss = 0.0
        correct = 0.0
        nb_test = 0
        for num_batch, batch_data in enumerate(data.test):
            x, y = batch_data
            x, y = x.squeeze().to(device), y.squeeze().to(device)
            if dataset == 'MNIST':
                x = torch.unsqueeze(x, dim=1)
            logits = F.log_softmax(net(x), dim=1)
            test_loss += loss_function_task(logits, y).item()
            _, y_pred = logits.max(1)
            correct += y_pred.eq(y).sum().item()
            nb_test += y.size(0)
        test_loss = test_loss / num_batch
        test_acc = correct / nb_test * 100
        logger.info(f"Test loss: {test_loss:.4f}, test accuracy: {test_acc}")
        torch.save(net.state_dict(), f'{base_dir}checkpoints/{experiment_name}.pt')
        writer.add_scalars('losses', {'train_loss': train_loss, 'test_loss': test_loss}, epoch)
        writer.add_scalars('accuracy', {'train_acc': train_acc, 'test_acc': test_acc}, epoch)
        writer.add_scalars('train_loss_task', {'train_loss_task': train_loss_task}, epoch)
        writer.add_scalars('train_loss_sc', {'train_loss_sc': train_loss_sc}, epoch)
    writer.close()


if __name__ == "__main__":
    flags.DEFINE_string("dataset", 'CIFAR10', "CIFAR10, MNIST, and TinyImageNet are supported as dataset.")
    flags.DEFINE_integer("batch_size", 64, "Training batch_size.")
    flags.DEFINE_integer("l_r", 0.001, "Initial learning rate.")
    flags.DEFINE_integer("epochs", 1000, "Number of training epochs.")
    flags.DEFINE_string("experiment_name", None, "Experiment name.")
    flags.DEFINE_string("base_dir", './', "The directory in which to save the trained model and data dictionary.")
    flags.DEFINE_string("sc_path", None, "Path to the trained sparse coder.")
    app.run(main)
