# Creates dataset for DL-CPP to replicate playground model
import csv
import argparse
import dataset
import random
import matplotlib.pyplot as plt
import numpy as np

TYPES = {
    'two_gaussians': dataset.classify_two_gauss_data,
    'spiral': dataset.classify_spiral_data,
    'circle': dataset.classify_circle_data,
    'xor': dataset.classify_xor_data
}

NUM_SAMPLES = 500

if __name__ == '__main__':

    print('Dataset Creator')

    parser = argparse.ArgumentParser(description='Dataset Creator')
    parser.add_argument('--type', type=str, default='two_gaussians', help='Type of dataset')
    parser.add_argument('--plot',  help='Plot the dataset', action='store_true')

    args = parser.parse_args()

    OUTPUT = f'./{args.type}_dataset.csv'

    with open(OUTPUT, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'label'])

        data = TYPES[args.type](NUM_SAMPLES)
        blue = []
        orange = []

        random.shuffle(data)

        for row in data:
            row = [round(row[0], 3), round(row[1], 3), row[2]]
            writer.writerow(row)
            if row[2] == 1:
                blue.append(row)
            else:
                orange.append(row)

        colored_data = [blue, orange]
        

        if args.plot:
            colors = ['blue', 'orange']

            xlims = plt.xlim()
            ylims = plt.ylim()

            x_min, x_max = -6, 6
            y_min, y_max = -6, 6

            xx, yy = np.meshgrid(np.arange(x_min, y_max, 0.2), np.arange(y_min, y_max, 0.2))
            fig, ax = plt.subplots()
            plt.tight_layout()
            ax.contourf(xx, yy, np.array([[1 if np.random.rand() > 0.5 else -1 for _ in range(len(xx[0]))] for _ in range(len(xx))]), alpha=0.5, colors=['#ADD8E6', '#FFA07A'])

            for i in range(2):
                ax.scatter([row[0] for row in colored_data[i]], [row[1] for row in colored_data[i]], color=colors[i])

            plt.show()

        