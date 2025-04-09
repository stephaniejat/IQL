# # Filters for the runs you want to include in the plot
# import wandb
# import matplotlib.pyplot as plt
# import numpy as np

# # Initialize API client
# api = wandb.Api()

# # Define entity and project
# entity = 'ayakayal'
# project = 'IQL backup'

# # Filters for the runs you want to include in the plot
# filters = {
#     'summary_metrics.alpha_functions': 0.01 #was 0.01
# }

# # Fetch runs matching the conditions for both algorithms
# runs = api.runs(f"{entity}/{project}", filters=filters)
# # Data storage
# data = {'GroupInvariantKernel()': [], 'RBF(length_scale=1)': []}
# episode_numbers = {'GroupInvariantKernel()': [], 'RBF(length_scale=1)': []}

# # Extract runs by kernel type
# for run in runs:
#     # print('run.summary',run.summary)
#     # print('run.',run.history)
#     #print(run.scan_history())
#     kernel_type = run.summary.get('kernel type', None)  # Fetch kernel_type from summary
#     # print('kernel_type:', kernel_type)
  
#     if kernel_type in data:
#         # Fetch history without relying on pandas
#         #history = run.history(keys=['Episode_Regret'], x_axis='_step')
        
#         history = run.history(keys=['Episode_Regret'], pandas=False, samples=1000)
#         # Extract Episode_Regret values
#         # print(history)
#         episode_regret = [entry['Episode_Regret'] for entry in history if 'Episode_Regret' in entry]
#         # print('ep regret len',len(episode_regret))
#         episode_number = [entry['_step'] for entry in history if '_step' in entry]
#         data[kernel_type].append(episode_regret)
#         episode_numbers[kernel_type].append(episode_number)
#         #print('ep nbs',episode_numbers[kernel_type])


# print('num runs invariant',len(data['GroupInvariantKernel()']))
# print('num runs standard',len(data['RBF(length_scale=1)']))
# #print('len data sample',data['GroupInvariantKernel()'][0])
# print('ep numbers 1',len(episode_numbers['GroupInvariantKernel()'][0]))
# print('ep numbers 2',len(episode_numbers['GroupInvariantKernel()'][1]))
# # Initialize plot
# # Define a mapping for better labels
# label_mapping = {
#     'GroupInvariantKernel()': 'Invariant Kernel',
#     'RBF(length_scale=1)': 'Standard RBF Kernel'
# }
# ##################################with running average ###################################
# # def running_average_wandb(data, smoothing_factor):
# #     half_window = smoothing_factor // 2
# #     smoothed = []
# #     for i in range(len(data)):
# #         # Define the window
# #         start = max(0, i - half_window)
# #         end = min(len(data), i + half_window + 1)
# #         # Compute the mean within the window
# #         smoothed.append(np.mean(data[start:end]))
# #     return np.array(smoothed)
# #     # Initialize plot
# # plt.figure(figsize=(10, 6))

# # for kernel_type, regrets_list in data.items():
# #     # Compute the average episodic regret over the runs
# #     mean_regrets = np.mean(regrets_list, axis=0)
# #     mean_episode_numbers = np.mean(episode_numbers[kernel_type], axis=0)

# #     # Apply running average smoothing
# #     smoothed_regrets = running_average_wandb(mean_regrets, smoothing_factor=10)
# #     smoothed_episodes = mean_episode_numbers  # No need to truncate episode numbers

# #     # Compute standard error (optional, for error shading)
# #     std_error = np.std(regrets_list, axis=0) / np.sqrt(len(regrets_list))
# #     smoothed_std_error = running_average_wandb(std_error, smoothing_factor=10)

# #     # Use the mapped label for the plot
# #     label = label_mapping.get(kernel_type, kernel_type)
# #     plt.plot(smoothed_episodes, smoothed_regrets, label=f'{label}', linewidth=2)

# #     # Add error shading if desired
# #     plt.fill_between(
# #         smoothed_episodes,
# #         smoothed_regrets - smoothed_std_error,
# #         smoothed_regrets + smoothed_std_error,
# #         alpha=0.2
# #     )

# # # Configure plot
# # plt.xlabel("Episode Number", fontsize=25)
# # plt.ylabel("Episodic Regret", fontsize=25)
# # plt.legend(fontsize=20)
# # plt.grid(False)
# # plt.tight_layout()

# # # Save and display plot
# # plt.savefig('smoothed_plot_wandb.png')



# ################################################ without averaging #########################
# # plt.figure(figsize=(10, 6))
# # for kernel_type, regrets_list in data.items():
# #     # Compute the average episodic regret over the 21 runs
# #     mean_regrets = np.mean(regrets_list, axis=0)
# #     # Get the corresponding episode numbers for the x-axis
# #     mean_episode_numbers = np.mean(episode_numbers[kernel_type], axis=0)
# #     # Use the mapped label for the plot
# # # Compute standard error (optional, for error shading)
# #     std_error = np.std(regrets_list, axis=0) / np.sqrt(len(regrets_list))
# #     label = label_mapping.get(kernel_type, kernel_type)  # Default to kernel_type if no mapping
# #     plt.plot(mean_episode_numbers, mean_regrets, label=f'{label}', linewidth=2)

# #     plt.fill_between(mean_episode_numbers,
# #                      mean_regrets - std_error,
# #                      mean_regrets + std_error,
# #                      alpha=0.2)
# # # Configure plot
# # #plt.title("Episodic Regret vs. Episode Number", fontsize=14)
# # plt.xlabel("Episode Number", fontsize=25)
# # plt.ylabel("Episodic Regret", fontsize=25)
# # plt.legend(fontsize=20)
# # plt.grid(False)
# # plt.tight_layout()
# # # Show plot
# # plt.savefig('plot_wandb_nosmoothing.png')

# ########################################################CumulativeRegret######################################
# # plt.figure(figsize=(10, 6))

# # for kernel_type, regrets_list in data.items():
# #     print('kernel_type',kernel_type)
# #     print('regret_list',len(regrets_list))
# #     # Convert episodic regrets into cumulative regrets for each run
# #     cumulative_regrets_list = [np.cumsum(run_regret) for run_regret in regrets_list]
# #     print('cum regret list',len(cumulative_regrets_list))
    
# #     # Compute the mean cumulative regret over the runs
# #     mean_cumulative_regrets = np.mean(cumulative_regrets_list, axis=0)
# #     # Get the corresponding episode numbers for the x-axis
# #     mean_episode_numbers = np.mean(episode_numbers[kernel_type], axis=0)
    
# #     # Compute standard error for cumulative regret
# #     std_error = np.std(cumulative_regrets_list, axis=0) / np.sqrt(len(cumulative_regrets_list))
    
# #     # Use the mapped label for the plot
# #     label = label_mapping.get(kernel_type, kernel_type)  # Default to kernel_type if no mapping
# #     plt.plot(mean_episode_numbers, mean_cumulative_regrets, label=f'{label}', linewidth=2)
    
# #     # Optional: Add error shading
# #     plt.fill_between(mean_episode_numbers,
# #                      mean_cumulative_regrets - std_error,
# #                      mean_cumulative_regrets + std_error,
# #                      alpha=0.2)

# # # Configure plot
# # plt.xlabel("Episode Number", fontsize=25)
# # plt.ylabel("Cumulative Regret", fontsize=25)
# # plt.legend(fontsize=20)
# # plt.grid(False)
# # plt.tight_layout()

# # # Save the plot
# # plt.savefig('cumulative_regret_plot.png')
# ################################################# Cumulative regret with slicing #####################
# plt.figure(figsize=(10, 6))

# for kernel_type, regrets_list in data.items():
#     cumulative_regrets_list = [np.cumsum(run_regret)[:300] for run_regret in regrets_list]
    
#     # Compute the mean cumulative regret over the runs (only for the first 300 episodes)
#     mean_cumulative_regrets = np.mean(cumulative_regrets_list, axis=0)
#     # Get the corresponding episode numbers for the x-axis (slice to first 300)
#     mean_episode_numbers = np.mean([run[:300] for run in episode_numbers[kernel_type]], axis=0)
#     print('mean episode numbers', mean_episode_numbers)
    
#     # Compute standard error for cumulative regret
#     std_error = np.std(cumulative_regrets_list, axis=0) / np.sqrt(len(cumulative_regrets_list))
    
#     # Use the mapped label for the plot
#     label = label_mapping.get(kernel_type, kernel_type)  # Default to kernel_type if no mapping
#     plt.plot(mean_episode_numbers, mean_cumulative_regrets, label=f'{label}', linewidth=2)
    
#     # Optional: Add error shading
#     plt.fill_between(mean_episode_numbers,
#                      mean_cumulative_regrets - std_error,
#                      mean_cumulative_regrets + std_error,
#                      alpha=0.2)

# # Configure plot
# plt.xlabel("Episode Number", fontsize=25)
# plt.ylabel("Cumulative Regret", fontsize=25)
# plt.legend(fontsize=20)
# plt.grid(False)
# plt.tight_layout()

# # Save the plot
# plt.savefig('cumulative_regret_300_episodes.png')
   
###########################################################################


# Assume `data` is already populated from the previous step
# Each entry in `data[kernel]` is a list of episodic regrets from one run.

# Target number of samples
# target_samples = 300

# # Initialize plot
# plt.figure(figsize=(10, 6))

# for kernel_type, regrets_list in data.items():
#     # Determine the total length of the episode regret array (assuming all runs are the same length)
#     n_episodes = len(regrets_list[0])
    
#     # Calculate indices for downsampling
#     downsample_indices = np.linspace(0, n_episodes - 1, target_samples, dtype=int)
    
#     # Downsample each run's episodic regret to the selected indices
#     downsampled_regrets = np.array([np.array(regret)[downsample_indices] for regret in regrets_list])
    
#     # Compute the average episodic regret over the downsampled runs
#     mean_regrets = np.mean(downsampled_regrets, axis=0)
    
#     # Compute standard error (optional, for error shading)
#     std_error = np.std(downsampled_regrets, axis=0) / np.sqrt(len(downsampled_regrets))
    
#     # Plot the average episodic regret
#     plt.plot(mean_regrets, label=f'{kernel_type} (average)', linewidth=2)
    
#     # Add shading for standard error (optional)
#     plt.fill_between(range(target_samples),
#                      mean_regrets - std_error,
#                      mean_regrets + std_error,
#                      alpha=0.2)

# # Configure plot
# plt.title("Episodic Regret vs. Episode Number (Downsampled)", fontsize=14)
# plt.xlabel("Episode Number", fontsize=12)
# plt.ylabel("Episodic Regret", fontsize=12)
# plt.legend(fontsize=10)
# plt.grid()
# plt.tight_layout()
###########################################################################
# Filters for the runs you want to include in the plot
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Initialize API client
api = wandb.Api()

# Define entity and project
entity = 'ayakayal'
project = 'IQL backup'

# Filters for the runs you want to include in the plot
filters = {
    'summary_metrics.alpha_functions': 0.001 #was 0.01
}

# Fetch runs matching the conditions for both algorithms
runs = api.runs(f"{entity}/{project}", filters=filters)
# Data storage
data = {'GroupInvariantKernel()': [], 'RBF(length_scale=1)': []}
episode_numbers = {'GroupInvariantKernel()': [], 'RBF(length_scale=1)': []}

# Extract runs by kernel type
for run in runs:
    # print('run.summary',run.summary)
    # print('run.',run.history)
    #print(run.scan_history())
    kernel_type = run.summary.get('kernel type', None)  # Fetch kernel_type from summary
    # print('kernel_type:', kernel_type)
  
    if kernel_type in data:
        # Fetch history without relying on pandas
        #history = run.history(keys=['Episode_Regret'], x_axis='_step')
        
        history = run.history(keys=['Episode_Rewards'], pandas=False, samples=1000)
        # Extract Episode_Regret values
        # print(history)
        episode_regret = [entry['Episode_Rewards'] for entry in history if 'Episode_Rewards' in entry]
        # print('ep regret len',len(episode_regret))
        episode_number = [entry['_step'] for entry in history if '_step' in entry]
        data[kernel_type].append(episode_regret)
        episode_numbers[kernel_type].append(episode_number)
        #print('ep nbs',episode_numbers[kernel_type])


print('num runs invariant',len(data['GroupInvariantKernel()']))
print('num runs standard',len(data['RBF(length_scale=1)']))
#print('len data sample',data['GroupInvariantKernel()'][0])
print('ep numbers 1',len(episode_numbers['GroupInvariantKernel()'][0]))
print('ep numbers 2',len(episode_numbers['GroupInvariantKernel()'][1]))
# Initialize plot
# Define a mapping for better labels
label_mapping = {
    'GroupInvariantKernel()': 'Invariant Kernel',
    'RBF(length_scale=1)': 'Standard RBF Kernel'
}

################################################# Cumulative regret with slicing #####################
plt.figure(figsize=(10, 6))

for kernel_type, regrets_list in data.items():
    cumulative_regrets_list = [np.cumsum(run_regret)[:300] for run_regret in regrets_list]
    
    # Compute the mean cumulative regret over the runs (only for the first 300 episodes)
    mean_cumulative_regrets = np.mean(cumulative_regrets_list, axis=0)
    # Get the corresponding episode numbers for the x-axis (slice to first 300)
    mean_episode_numbers = np.mean([run[:300] for run in episode_numbers[kernel_type]], axis=0)
    print('mean episode numbers', mean_episode_numbers)
    
    # Compute standard error for cumulative regret
    std_error = np.std(cumulative_regrets_list, axis=0) / np.sqrt(len(cumulative_regrets_list))
    
    # Use the mapped label for the plot
    label = label_mapping.get(kernel_type, kernel_type)  # Default to kernel_type if no mapping
    plt.plot(mean_episode_numbers, mean_cumulative_regrets, label=f'{label}', linewidth=2)
    
    # Optional: Add error shading
    plt.fill_between(mean_episode_numbers,
                     mean_cumulative_regrets - std_error,
                     mean_cumulative_regrets + std_error,
                     alpha=0.2)

# Configure plot
plt.xlabel("Episode Number", fontsize=25)
plt.ylabel("Cumulative Returns", fontsize=25)
plt.legend(fontsize=20)
plt.grid(False)
plt.tight_layout()

# Save the plot
plt.savefig('cumulative_returns_300_episodes.png')
   
###########################################################################


# Assume `data` is already populated from the previous step
# Each entry in `data[kernel]` is a list of episodic regrets from one run.

# Target number of samples
# target_samples = 300

# # Initialize plot
# plt.figure(figsize=(10, 6))

# for kernel_type, regrets_list in data.items():
#     # Determine the total length of the episode regret array (assuming all runs are the same length)
#     n_episodes = len(regrets_list[0])
    
#     # Calculate indices for downsampling
#     downsample_indices = np.linspace(0, n_episodes - 1, target_samples, dtype=int)
    
#     # Downsample each run's episodic regret to the selected indices
#     downsampled_regrets = np.array([np.array(regret)[downsample_indices] for regret in regrets_list])
    
#     # Compute the average episodic regret over the downsampled runs
#     mean_regrets = np.mean(downsampled_regrets, axis=0)
    
#     # Compute standard error (optional, for error shading)
#     std_error = np.std(downsampled_regrets, axis=0) / np.sqrt(len(downsampled_regrets))
    
#     # Plot the average episodic regret
#     plt.plot(mean_regrets, label=f'{kernel_type} (average)', linewidth=2)
    
#     # Add shading for standard error (optional)
#     plt.fill_between(range(target_samples),
#                      mean_regrets - std_error,
#                      mean_regrets + std_error,
#                      alpha=0.2)

# # Configure plot
# plt.title("Episodic Regret vs. Episode Number (Downsampled)", fontsize=14)
# plt.xlabel("Episode Number", fontsize=12)
# plt.ylabel("Episodic Regret", fontsize=12)
# plt.legend(fontsize=10)
# plt.grid()
# plt.tight_layout()



