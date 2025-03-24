import os
import concurrent.futures

# Define experiment configurations as a list of command strings
commands = [
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 1 --optim_botorch 1 --kernel invariant_kernel --iterations 1000 > GP_logs/log_inv_1_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 2 --optim_botorch 1 --kernel invariant_kernel --iterations 1000 > GP_logs/log_inv_2_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 3 --optim_botorch 1 --kernel invariant_kernel --iterations 1000 > GP_logs/log_inv_3_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 5 --optim_botorch 1 --kernel invariant_kernel --iterations 1000 > GP_logs/log_inv_5_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 6 --optim_botorch 1 --kernel invariant_kernel --iterations 1000 > GP_logs/log_inv_6_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 7 --optim_botorch 1 --kernel invariant_kernel --iterations 1000 > GP_logs/log_inv_7_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 8 --optim_botorch 1 --kernel invariant_kernel --iterations 1000 > GP_logs/log_inv_8_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 9 --optim_botorch 1 --kernel invariant_kernel --iterations 1000 > GP_logs/log_inv_9_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.5 --noise_reg 0.1 --seed 10 --optim_botorch 1 --kernel invariant_kernel --iterations 1000 > GP_logs/log_inv_10_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 1 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_1_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 2 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_2_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 3 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_3_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 4 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_4_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 5 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_5_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 6 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_6_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 7 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_7_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 8 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_8_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 9 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_9_h_30.txt",
"python3 -u KRVI_algo_test_rotated_invariant.py --beta 0.01 --len_scale 0.1 --noise_reg 0.001 --seed 10 --optim_botorch 0 --kernel RBF --iterations 1000 > GP_logs/log_base_10_h_30.txt"
]

# Loop through each command and execute it
def run_command(cmd):
    print(f"Running: {cmd}")
    os.system(cmd)

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(run_command, commands)

print("All experiments completed.")
