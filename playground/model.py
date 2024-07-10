import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.optim as optim
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
    'cross_entropy': torch.nn.CrossEntropyLoss()
}

NUM_SAMPLES = 500

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
                self.labels.append(int(row[2]))

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
    
def train(model, data_loader, epochs, lr, loss):
    loss_fn = LOSS[loss]
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for i, data in enumerate(data_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')

def test(model, data_loader, loss):
    loss_fn = LOSS[loss]
    total = 0
    correct = 0

    for data in data_loader:
        inputs, labels = data
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

        print(f'Loss: {loss.item()}')

    print(f'Accuracy: {correct / total}')


if __name__ == '__main__':
    
    print('Playground')

    parser = argparse.ArgumentParser(description='Playground')
    parser.add_argument('--type', type=str, default='two_gaussians', help='Type of dataset')
    parser.add_argument('--h_sizes', type=int, nargs='+', default=[2, 4, 1], help='Hidden layers sizes')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
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

    model = Model(args.h_sizes, args.activation)

    train(model, train_loader, args.epochs, args.lr, args.loss)
    test(model, test_loader, args.loss)
