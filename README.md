# Stochastic Gradient Descent Optimizer Aided by High-Order Numerical Methods
This is an offitial implementation of the paper "Stochastic Gradient Descent Optimizer Aided by High-Order Numerical Methods". 

## Description
The code for the learnable coefficients is placed in the `parameter` folder.

The `HSGD.py` file is the optimizer that we proposed in the paper.

## Research architecture of the paper.
![HSGD architecture](./image/HSGD_arch.png)

## A quick look at the algorithm
![HSGD algorithm](./image/HSGD_algo.png)

## Useage in PyTorch
Simply put "HSGD.py" in your main file path, and add this line in the head of your training script:

``` from HSGD import HSGD```

Change the optimizer as

``` optimizer = HSGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, alpha=args.alpha, beta=args.beta) ```

And in you training code, use `optim.step(step=step)` instead of `optim.step()`

Run your code. 
