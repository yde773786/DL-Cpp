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

## Get Started

The major working parts that need to be puzzled together to design and create a model and test/train workflow using `DL-CPP` are as listed:

- `Model`: The architecture of the model created. The template.
- `Dataset`: Wrapper for raw data
- `DataLoader`: Wrapper for obtaining train/test batches or "loading" from `dataset`
- `Config`: Design the parameters required and how they are read. This needs to be provided when creating a `model`

### Example Design
`Perceptron` workflow using `Playground`

- `Model`: `Perceptron`
- `Dataset`: `PlaygroundDataset`
- `DataLoader`: `PlaygroundDataLoader`
- `Config`: [CFG](/config/perceptron/perceptron_scalar.cfg)

### Build
`DL-CPP`
```
cmake --build . --target clean; make;
```

`Autodiff` Tests
```
cd units/autodiff/tests && cmake --build . --target clean; make; cd ../../../
```

### Run
`DL-CPP` Suggested run command
```
./dl_cpp <cfg> > log 2> debug
```

`Autodiff` Tests
```
./units/autodiff/tests/test_autodiff
```
