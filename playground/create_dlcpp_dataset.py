# Creates dataset for DL-CPP to replicate playground model
import csv
import argparse
import dataset
import random
import plot_data

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
            writer.writerow(row)

    if args.plot:   
        plot_data.plot_data(OUTPUT)

        