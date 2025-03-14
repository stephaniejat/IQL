import os

# Define experiment configurations as a list of command strings
commands = [
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 1 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 2 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 3 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 5 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 6 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 7 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 8 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 9 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 10 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 11 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 12 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 13 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 14 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 15 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 16 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 17 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 18 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 19 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 20 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 21 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 22 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 23 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 24 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 25 --optim_botorch 1 --kernel invariant_kernel --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 1 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 2 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 3 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 4 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 5 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 6 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 7 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 8 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 9 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 10 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 11 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 12 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 13 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 14 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 15 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 16 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 17 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 18 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 19 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 20 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 21 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 22 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 23 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 24 --optim_botorch 0 --kernel RBF --iterations 1000",
"python KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 25 --optim_botorch 0 --kernel RBF --iterations 1000"
 
]

# Loop through each command and execute it
for cmd in commands:
    print(f"Running: {cmd}")
    os.system(cmd)

print("All experiments completed.")
