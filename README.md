# IQL
Kernelized Q-learning with invariances (IQL) is an implementation of LSVI algorithm designed to incorporate invariance properties into the kernel function. This can improve sample efficiency and generalization, especially in structured environments.

## Synthetic Experiments
To run synthetic experiments:

```bash
python script.py
```
It will run the KRVI algorithm with the invariant kernel on the synthetically generated reward and transition dynamics functions, for T=1000 episodes, using the UCB coefficient $\beta$= 0.1 and averaging over NUM_RUNS=20 runs. You can change these parameters in script.py
## Frozen Lake - Fixed Layout
To run experiments on the Frozen Lake environment with a fixed layout:

```bash
python KRVI_algo_test_rotated_invariant_fulloptim.py
```
You can specify hyperparparameters as arguments. 

We used the following hyperparameters in our experiments:

For the invariant kernel:
```bash
 --kernel invariant_kernel --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --optim_botorch 1 --iterations 1500 
 ```

For the RBF kernel:
```bash
 --kernel RBF --beta 0.01 --len_scale 0.1 --noise_reg 0.1 --optim_botorch 1 --iterations 1500
 ```
## Frozen Lake - Random Layout
To run experiments on the Frozen Lake environment with random layouts:


```bash
python KRVI_algo_test_rotated_invariant_fulloptim_eval.py
```
We used the following hyperparameters in our experiments: 

For the invariant kernel: 
```bash
--kernel invariant_kernel --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --optim_botorch 1 --iterations 2000
```

For the RBF kernel:
```bash
 --kernel RBF --beta 0.01 --len_scale 1 --noise_reg 0.1 --optim_botorch 1 --iterations 2000
```