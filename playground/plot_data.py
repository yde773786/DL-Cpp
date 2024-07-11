from matplotlib import pyplot as plt
import csv
import numpy as np

"""
Plot the data from a csv file

If a function is provided, it will be used to plot the decision region. 
An output of -1 is colored orange, an output of 1 is colored blue.
"""
def plot_data(file_str, function=None):

    blue = []
    orange = []

    with open(file_str, mode='r') as file:
        csvreader = csv.reader(file)
        for i, row in enumerate(csvreader):

            if i == 0:
                continue

            row = [float(row[0]), float(row[1]), int(row[2])]
            if row[2] == 1:
                blue.append(row)
            else:
                orange.append(row)

    colored_data = [blue, orange]

    colors = ['blue', 'orange']
    _, ax = plt.subplots()
    plt.tight_layout()

    if function:
        x_min, x_max = -6, 6
        y_min, y_max = -6, 6

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.2),
            np.arange(y_min, y_max, 0.2)
        )

        Z = function(xx, yy)
        ax.contourf(xx, yy, Z, alpha=0.5)

    for i in range(2):
        ax.scatter([row[0] for row in colored_data[i]], [row[1] for row in colored_data[i]], color=colors[i])

    plt.show()