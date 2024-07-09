# Simplified version with required functionality of the dataset generation in the tensorflow playground
# https://github.com/tensorflow/playground/blob/master/src/dataset.ts

# Dataset generation currently supports the following types:
# - Two Gaussians
# - Spiral
# - Circle
# - XOR

# No support for noise

import random
import math

def classify_two_gauss_data(num_samples):
    data = []

    def generate_gauss(mean, variance, label):
        for _ in range(num_samples / 2):
            x = random.gauss(mean, variance)
            y = random.gauss(mean, variance)
            data.append([x, y, label])

    # Gaussian with positive examples.
    generate_gauss(2, 2, 1)
    # Gaussian with negative examples.
    generate_gauss(-2, -2, -1)

    return data

def classify_spiral_data(num_samples):
    data = []

    def generate_spiral(deltaT, label):
        for i in range(num_samples / 2):
            r = i / num_samples * 5
            t = 1.75 * i / num_samples * 2 * math.pi + deltaT
            x = r * math.sin(t)
            y = r * math.cos(t)
            data.append([x, y, label])

    # Positive examples.
    generate_spiral(0, 1)
    # Negative examples.
    generate_spiral(math.pi, -1)

    return data


def classify_circle_data(num_samples):
    data = []

    radius = 5

    def generate_circle_label(p, center):
        dx = p[0] - center[0]
        dy = p[1] - center[1]
        
        return 1 if dx ** 2 + dy ** 2 < (radius * 0.5) ** 2 else -1
    
    for i in range(num_samples / 2):
        r = random.uniform(0, radius * 0.5)
        angle = random.uniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)

        label = generate_circle_label([x, y], [0, 0])
        data.append([x, y, label])

    return data

def classify_xor_data(num_samples):
    data = []

    def generate_xor_label(p):
        return 1 if p[0] * p[1] >= 0 else -1
    
    for i in range(num_samples):
        x = random.uniform(-5, 5)
        padding = 0.3
        
        if x > 0:
            x += padding
        else:
            x -= padding

        y = random.uniform(-5, 5)
        if y > 0:
            y += padding
        else:
            y -= padding

        label = generate_xor_label([x, y])
        data.append([x, y, label])

    return data