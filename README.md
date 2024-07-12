# DL-Cpp

Popular Neural Network models implemented in C++ 

## Aim
- Create Deep Learning models with simple `cfg` ([`libconfig++`](https://github.com/hyperrealm/libconfig)) files.
- Modularize the creation of models to enable more designing (through `cfg`) then programming.
- Self-learning and Experimentation on model architectures, scalar and vectorized code (scalar vs `CUDA` vs `SIMD intrinsics`) performance comparison, autodiff etc.

We use [Playground](https://github.com/yde773786/DL-Cpp/tree/main/playground) for evaluation of correctness and performance.

## Create a dataset

### Command

`python3 create_dlcpp_dataset.py [-h] [--type TYPE] [--plot]`

```
  -h, --help   show this help message and exit
  --type TYPE  Type of dataset
  --plot       Plot the dataset
```

### Example
```
python3 create_dlcpp_dataset.py --plot --type xor
```
<img src="https://github.com/user-attachments/assets/09d9938d-8734-4b65-88dc-041164d6a17b" width="200" height="200" />

## Use `PyTorch` for playground experiment

### Command
```
model.py [-h] [--type TYPE] [--layer-sizes LAYER_SIZES [LAYER_SIZES ...]]
                [--activation ACTIVATION] [--batch-size BATCH_SIZE] [--plot-loss]
                [--plot-data] [--epochs EPOCHS] [--lr LR] [--loss LOSS]
                [--split SPLIT]
```

### Example
Use a neural network with:
- 3 hidden layers with 6, 2, 2 neurons respectively
- 150 epochs training
- batch size 10
- plot the decision boundary/area

```
python3 model.py --layer-sizes 2 6 2 2 1  --batch-size 10 --epochs 150 --type xor --plot-data
```
Before Training:

<img src="https://github.com/user-attachments/assets/864afa89-931d-418b-9c4a-029bba22b695" width="200" height="200" />

After Training:

<img src="https://github.com/user-attachments/assets/74de2d06-d58a-45e7-a896-06acf2f3590a" width="200" height="200" />

## Use `DL-CPP` for playground experiment
TODO
