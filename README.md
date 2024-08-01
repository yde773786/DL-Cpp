# DL-Cpp

C++ lightweight library enabling the creation of Machine Learning models using configuration files

## Features Supported
- [x] Create [Playground](./playground), a benchmarker to measure correctness and performance of ML frameworks vs `PyTorch`.
- [x] Support cross-compatible serialization of weights between `PyTorch` and `DL-CPP`
- [x] Provide framework for modular creation of Deep Learning models, and ablility to represent with simple `cfg` ([`libconfig++`](https://github.com/hyperrealm/libconfig)) files.
- [x] `Perceptron` `cfg` support pre-packaged
- [x] Support for lightweight automatic differentiation engine. [Reverse mode autodiff library](./units/autodiff) for general usage as well as specifically backpropogation in `DL-Cpp`

## Features under progress
- [ ] Use `pybind11` to expose `Playground` to `DL-CPP` for graphing purposes
- [ ] Support for `CUDA` and using `SIMD intrinsics` instead of scalar code for forward and backpropogation
- [ ] pre-packaged `cfg` support for `FNN`, `CNN` and `RNN`



## Create a dataset

### Command

`python3 create_dlcpp_dataset.py [-h] [--type TYPE] [--plot]`

### Example
```
python3 create_dlcpp_dataset.py --plot --type xor
```
<img src="https://github.com/user-attachments/assets/09d9938d-8734-4b65-88dc-041164d6a17b" width="200" height="200" />

## Use `PyTorch` for playground experiment

### Command
```
model.py [-h] [--type TYPE] [--layer-sizes LAYER_SIZES [LAYER_SIZES ...]] [--activation ACTIVATION] [--batch-size BATCH_SIZE] [--plot-loss] [--plot-data]
                [--epochs EPOCHS] [--use-pretrained USE_PRETRAINED] [--save-weight] [--lr LR] [--loss LOSS] [--split SPLIT]
```

### Example
Use a neural network with:
- 3 hidden layers with 6, 2, 2 neurons respectively
- 150 epochs training
- batch size 10
- plot the decision boundary/area
- save weights in a cross compatible format (works with `pyTorch` and `DL-CPP`)

```
python3 model.py --layer-sizes 2 6 2 2 1  --batch-size 10 --epochs 150 --type xor --plot-data
```
Before Training:

<img src="https://github.com/user-attachments/assets/864afa89-931d-418b-9c4a-029bba22b695" width="200" height="200" />

After Training:

<img src="https://github.com/user-attachments/assets/74de2d06-d58a-45e7-a896-06acf2f3590a" width="200" height="200" />

## Use `DL-CPP` for playground experiment

Use example config files available at [Config](./config)

### Command
```
./dl_cpp <config_file>
```
