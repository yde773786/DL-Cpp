from re import A
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
import plot_data
import matplotlib.pyplot as plt
import argparse
import csv

TYPES = {
    'two_gaussians': 'two_gaussians_dataset.csv',
    'spiral': 'spiral_dataset.csv',
    'circle': 'circle_dataset.csv',
    'xor': 'xor_dataset.csv'
}

ACTIVATION = {
    'relu': torch.nn.ReLU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid()
}

LOSS = {
    'mse': torch.nn.MSELoss(),
    'cross_entropy': torch.nn.CrossEntropyLoss(),
    'l1-loss': torch.nn.L1Loss()
}

NUM_SAMPLES = 500
model = None

class PointsDataset(Dataset):

    def __init__(self, type) -> None:
        super().__init__()
        self.data = []
        self.labels = []
        with open(f'./{TYPES[type]}', mode='r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.data.append(torch.tensor([float(row[0]), float(row[1])], dtype=torch.float32))
                self.labels.append(torch.tensor(float(row[2]), dtype=torch.float32))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    

class Model(torch.nn.Module):

    def __init__(self, nn_sizes, activation) -> None:
        super().__init__()
        modules = []

        for i in range(len(nn_sizes) - 2):
            modules.append(torch.nn.Linear(nn_sizes[i], nn_sizes[i + 1]))
            modules.append(ACTIVATION[activation])

        modules.append(torch.nn.Linear(nn_sizes[-2], nn_sizes[-1]))
        modules.append(ACTIVATION['tanh'])
        
        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
    
def train(model, data_loader, epochs, lr, loss, plot_loss):
    loss_fn = LOSS[loss]
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_vals = []
    
    for epoch in range(epochs):
        for i, data in enumerate(data_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels.view(-1, 1))

            loss.backward()
            optimizer.step()

            if plot_loss:
                loss_vals.append(loss.item())

            print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')

    if plot_loss:
        plt.plot(loss_vals)
        plt.title('Training Loss')
        plt.show()

def test(model, data_loader, loss_func, plot_loss):
    loss_fn = LOSS[loss_func]
    total = 0
    correct = 0

    loss_vals = []

    for data in data_loader:
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels.view(-1, 1))

        total += labels.size(0)
        predicted = 1 if outputs.item() > 0 else -1

        correct += (predicted == labels).sum().item()

        if plot_loss:
            loss_vals.append(loss.item())

        print(f'Loss: {loss.item()}')

    print(f'Accuracy: {correct / total}')

    if plot_loss:
        plt.plot(loss_vals)
        plt.title('Training Loss')
        plt.show()

def plot_Z_func(x_i, y_i):
    assert model is not None, 'Model is not defined'
    predicted = 1 if model(torch.tensor([x_i, y_i], dtype=torch.float32)).item() > 0 else -1

    return predicted

if __name__ == '__main__':
    
    print('Playground')

    parser = argparse.ArgumentParser(description='Playground')
    parser.add_argument('--type', type=str, default='two_gaussians', help='Type of dataset')
    # Default model is a perceptron (2 input, 1 output)
    parser.add_argument('--layer-sizes', type=int, nargs='+', default=[2, 1], help='Hidden layers sizes')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--plot-loss', help='Plot loss', action='store_true')
    parser.add_argument('--plot-data', help='Plot data', action='store_true')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--use-pretrained', type=str, help='Use pretrained weights')
    parser.add_argument('--save-weight', help='Save model weights, cross compatible with DL-CPP', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='Loss function')
    parser.add_argument('--split', type=float, default=0.8, help='Train:test split ratio (x:(1-x))')

    args = parser.parse_args()
    dataset = PointsDataset(args.type)
    
    indices = list(range(len(dataset)))
    split = int(args.split * len(dataset))
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_sampler = SequentialSampler(train_indices)
    test_sampler = SequentialSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, sampler=test_sampler)

    model = Model(args.layer_sizes, args.activation)

    if args.plot_data:
        plot_data.plot_data(f'./{TYPES[args.type]}', plot_Z_func)

    if args.use_pretrained:
        load_arr = np.fromfile(args.use_pretrained)

        print(load_arr)
        shape, old_shape = 0, 0
        
        for param_tensor in model.state_dict():
            state, val = param_tensor, model.state_dict()[param_tensor]
            print(f'Loading State: {state}')

            # Get dimensions of the tensor as tuple
            shape = tuple(val.shape)

            # Load the weights and biases into the model
            model.state_dict()[param_tensor].copy_(torch.tensor(load_arr[old_shape:old_shape + np.prod(shape)]).view(shape))
            old_shape = old_shape + np.prod(shape) 

            print(model.state_dict()[param_tensor])
    else:
        train(model, train_loader, args.epochs, args.lr, args.loss, args.plot_loss)

    test(model, test_loader, args.loss, args.plot_loss)

    if args.plot_data:
        plot_data.plot_data(f'./{TYPES[args.type]}', plot_Z_func)

    if args.save_weight:
        save_arr = np.array([])
        save_file = open(f'wts_{args.type}_{args.layer_sizes}.wt', 'wb')

        for param_tensor in model.state_dict():
            state, val = param_tensor, model.state_dict()[param_tensor]
            print(f'Saving State: {state}')
            print(model.state_dict()[param_tensor])
            save_arr = np.append(save_arr, val.data.numpy().flatten())
            
        save_arr.tofile(save_file)

        