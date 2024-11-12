import numpy as np
import os
import matplotlib.pyplot as plt

run_path='experiment1'

picture_path='figures/RBF_kernel_all_algos.png'
benchmark_avg_regrets_runs = np.load(f'{run_path}/Benchmark Model_RBF_0.1_80/avg_regrets_runs.npy')
benchmark_std_regrets_runs = np.load(f'{run_path}/Benchmark Model_RBF_0.1_80/std_regrets_runs.npy')
benchmark_avg_regrets_mean = np.mean(benchmark_avg_regrets_runs, axis=0)
benchmark_std_regrets_std = np.sqrt(np.mean(np.square(benchmark_std_regrets_runs), axis=0))


# Load max variance data
max_variance_avg_regrets_runs = np.load(f'{run_path}/Greedy Max Variance_RBF_0.1_80/avg_regrets_runs.npy')
max_variance_std_regrets_runs = np.load(f'{run_path}/Greedy Max Variance_RBF_0.1_80/std_regrets_runs.npy')
max_variance_avg_regrets_mean = np.mean(max_variance_avg_regrets_runs, axis=0)
max_variance_std_regrets_std = np.sqrt(np.mean(np.square(max_variance_std_regrets_runs), axis=0))


# Load max variance data 2
max_variance2_avg_regrets_runs = np.load(f'{run_path}/Without generative model_RBF_0.1_80/avg_regrets_runs.npy')
max_variance2_std_regrets_runs = np.load(f'{run_path}/Without generative model_RBF_0.1_80/std_regrets_runs.npy')
max_variance2_avg_regrets_mean = np.mean(max_variance2_avg_regrets_runs, axis=0)
max_variance2_std_regrets_std = np.sqrt(np.mean(np.square(max_variance2_std_regrets_runs), axis=0))

# Load generative model
generative_model_avg_regrets_runs = np.load(f'{run_path}/Generative model_RBF_0.1_80/avg_regrets_runs.npy')
generative_model_std_regrets_runs = np.load(f'{run_path}/Generative model_RBF_0.1_80/std_regrets_runs.npy')
generative_model_avg_regrets_mean = np.mean(generative_model_avg_regrets_runs, axis=0)
generative_model_std_regrets_std = np.sqrt(np.mean(np.square(generative_model_std_regrets_runs), axis=0))


episode_list=[10, 20, 40, 80, 160]
# Plot average regret with standard deviation for both models
plt.errorbar(episode_list, benchmark_avg_regrets_mean, yerr=benchmark_std_regrets_std, fmt='o',color='b')
plt.plot(episode_list, benchmark_avg_regrets_mean, linestyle='-', marker='o', color='b', label='(Qiu et al., 2021)')  # Line connecting points
plt.errorbar(episode_list, max_variance_avg_regrets_mean, yerr=max_variance_std_regrets_std, fmt='o',color='r')
plt.plot(episode_list, max_variance_avg_regrets_mean, linestyle='-', marker='o', color='r', label='Greedy Max Variance')  # Line connecting points
plt.errorbar(episode_list, max_variance2_avg_regrets_mean, yerr=max_variance2_std_regrets_std, fmt='o', color='g')
plt.plot(episode_list, max_variance2_avg_regrets_mean, linestyle='-', marker='o', color='g', label='Without generative model')
plt.errorbar(episode_list, generative_model_avg_regrets_mean, yerr=generative_model_std_regrets_std, fmt='o', color='purple')
plt.plot(episode_list, generative_model_avg_regrets_mean, linestyle='-', marker='o', color='purple', label='With generative model')
# Configure matplotlib to use LaTeX for text rendering
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'  # You can add more packages if needed
plt.xlabel(r'$N$',fontsize=20)
plt.ylabel(r'$V^{\star}(s) -V^{\pi}(s)$',fontsize=20)
# plt.title('Average Regret with Standard Deviation')
plt.legend()
plt.savefig(picture_path)
plt.close()
# #############################################################################

picture_path='figures/Matern1.5_all_algos.png'


benchmark_avg_regrets_runs = np.load(f'{run_path}/Benchmark Model_Matern_smoothness_1.5_0.1_80/avg_regrets_runs.npy')
benchmark_std_regrets_runs = np.load(f'{run_path}/Benchmark Model_Matern_smoothness_1.5_0.1_80/std_regrets_runs.npy')
benchmark_avg_regrets_mean = np.mean(benchmark_avg_regrets_runs, axis=0)
benchmark_std_regrets_std = np.sqrt(np.mean(np.square(benchmark_std_regrets_runs), axis=0))


# Load max variance data
max_variance_avg_regrets_runs = np.load(f'{run_path}/Greedy Max Variance_Matern_smoothness_1.5_0.1_80/avg_regrets_runs.npy')
max_variance_std_regrets_runs = np.load(f'{run_path}/Greedy Max Variance_Matern_smoothness_1.5_0.1_80/std_regrets_runs.npy')
max_variance_avg_regrets_mean = np.mean(max_variance_avg_regrets_runs, axis=0)
max_variance_std_regrets_std = np.sqrt(np.mean(np.square(max_variance_std_regrets_runs), axis=0))


# Load max variance data 2
max_variance2_avg_regrets_runs = np.load(f'{run_path}/Without generative model_Matern_smoothness_1.5_0.1_80/avg_regrets_runs.npy')
max_variance2_std_regrets_runs = np.load(f'{run_path}/Without generative model_Matern_smoothness_1.5_0.1_80/std_regrets_runs.npy')
max_variance2_avg_regrets_mean = np.mean(max_variance2_avg_regrets_runs, axis=0)
max_variance2_std_regrets_std = np.sqrt(np.mean(np.square(max_variance2_std_regrets_runs), axis=0))

# Load generative model
generative_model_avg_regrets_runs = np.load(f'{run_path}/Generative model_Matern_smoothness_1.5_0.1_80/avg_regrets_runs.npy')
generative_model_std_regrets_runs = np.load(f'{run_path}/Generative model_Matern_smoothness_1.5_0.1_80/std_regrets_runs.npy')
generative_model_avg_regrets_mean = np.mean(generative_model_avg_regrets_runs, axis=0)
generative_model_std_regrets_std = np.sqrt(np.mean(np.square(generative_model_std_regrets_runs), axis=0))


episode_list=[10, 20, 40, 80, 160]
# Plot average regret with standard deviation for both models
plt.errorbar(episode_list, benchmark_avg_regrets_mean, yerr=benchmark_std_regrets_std, fmt='o',color='b')
plt.plot(episode_list, benchmark_avg_regrets_mean, linestyle='-', marker='o', color='b', label='(Qiu et al., 2021)')  # Line connecting points
plt.errorbar(episode_list, max_variance_avg_regrets_mean, yerr=max_variance_std_regrets_std, fmt='o',color='r')
plt.plot(episode_list, max_variance_avg_regrets_mean, linestyle='-', marker='o', color='r', label='Greedy Max Variance')  # Line connecting points
plt.errorbar(episode_list, max_variance2_avg_regrets_mean, yerr=max_variance2_std_regrets_std, fmt='o', color='g')
plt.plot(episode_list, max_variance2_avg_regrets_mean, linestyle='-', marker='o', color='g', label='Without generative model')
plt.errorbar(episode_list, generative_model_avg_regrets_mean, yerr=generative_model_std_regrets_std, fmt='o', color='purple')
plt.plot(episode_list, generative_model_avg_regrets_mean, linestyle='-', marker='o', color='purple', label='With generative model')

plt.xlabel(r'$N$',fontsize=20)
plt.ylabel(r'$V^{\star}(s) -V^{\pi}(s)$',fontsize=20)
# plt.title('Average Regret with Standard Deviation')
#plt.legend()
# Adjust the legend position to be at the bottom and center with a larger bounding box
plt.legend()
plt.savefig(picture_path)
plt.close()
# #######################################################################
picture_path='figures/Matern2.5_all_algos.png'


benchmark_avg_regrets_runs = np.load(f'{run_path}/Benchmark Model_Matern_smoothness_2.5_0.1_80/avg_regrets_runs.npy')
benchmark_std_regrets_runs = np.load(f'{run_path}/Benchmark Model_Matern_smoothness_2.5_0.1_80/std_regrets_runs.npy')
benchmark_avg_regrets_mean = np.mean(benchmark_avg_regrets_runs, axis=0)
benchmark_std_regrets_std = np.sqrt(np.mean(np.square(benchmark_std_regrets_runs), axis=0))


# Load max variance data
max_variance_avg_regrets_runs = np.load(f'{run_path}/Greedy Max Variance_Matern_smoothness_2.5_0.1_80/avg_regrets_runs.npy')
max_variance_std_regrets_runs = np.load(f'{run_path}/Greedy Max Variance_Matern_smoothness_2.5_0.1_80/std_regrets_runs.npy')
max_variance_avg_regrets_mean = np.mean(max_variance_avg_regrets_runs, axis=0)
max_variance_std_regrets_std = np.sqrt(np.mean(np.square(max_variance_std_regrets_runs), axis=0))


# Load max variance data 2
max_variance2_avg_regrets_runs = np.load(f'{run_path}/Without generative model_Matern_smoothness_2.5_0.1_80/avg_regrets_runs.npy')
max_variance2_std_regrets_runs = np.load(f'{run_path}/Without generative model_Matern_smoothness_2.5_0.1_80/std_regrets_runs.npy')
max_variance2_avg_regrets_mean = np.mean(max_variance2_avg_regrets_runs, axis=0)
max_variance2_std_regrets_std = np.sqrt(np.mean(np.square(max_variance2_std_regrets_runs), axis=0))

# Load generative model
generative_model_avg_regrets_runs = np.load(f'{run_path}/Generative model_Matern_smoothness_2.5_0.1_80/avg_regrets_runs.npy')
generative_model_std_regrets_runs = np.load(f'{run_path}/Generative model_Matern_smoothness_2.5_0.1_80/std_regrets_runs.npy')
generative_model_avg_regrets_mean = np.mean(generative_model_avg_regrets_runs, axis=0)
generative_model_std_regrets_std = np.sqrt(np.mean(np.square(generative_model_std_regrets_runs), axis=0))


episode_list=[10, 20, 40, 80, 160]
# Plot average regret with standard deviation for both models
plt.errorbar(episode_list, benchmark_avg_regrets_mean, yerr=benchmark_std_regrets_std, fmt='o',color='b')
plt.plot(episode_list, benchmark_avg_regrets_mean, linestyle='-', marker='o', color='b', label='(Qiu et al., 2021)')  # Line connecting points
plt.errorbar(episode_list, max_variance_avg_regrets_mean, yerr=max_variance_std_regrets_std, fmt='o',color='r')
plt.plot(episode_list, max_variance_avg_regrets_mean, linestyle='-', marker='o', color='r', label='Greedy Max Variance')  # Line connecting points
plt.errorbar(episode_list, max_variance2_avg_regrets_mean, yerr=max_variance2_std_regrets_std, fmt='o', color='g')
plt.plot(episode_list, max_variance2_avg_regrets_mean, linestyle='-', marker='o', color='g', label='Without generative model')
plt.errorbar(episode_list, generative_model_avg_regrets_mean, yerr=generative_model_std_regrets_std, fmt='o', color='purple')
plt.plot(episode_list, generative_model_avg_regrets_mean, linestyle='-', marker='o', color='purple', label='With generative model')

plt.xlabel(r'$N$',fontsize=20)
plt.ylabel(r'$V^{\star}(s) -V^{\pi}(s)$',fontsize=20)
# plt.title('Average Regret with Standard Deviation')
plt.legend()

plt.savefig(picture_path)
plt.close()