import matplotlib.pyplot as plt
import csv
import os
import numpy as np
from scipy.stats import gaussian_kde


# for i in os.listdir('consistency_analysis'):
#     csv_filename = f"consistency_analysis/{str(i)}"
#     new_csv_filename = f"consistency_analysis/plottable_{str(i)}"
#     with open(csv_filename, newline='') as infile, open(new_csv_filename, 'w', newline='') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)
#         writer.writerow(['Episode', 'Reward'])
#         for row in reader:
#             writer.writerow(row)


data = {}


def get_data(csv_filename, column):
    values = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            values.append(float(row[column]))
    return values



def convergence_check(csv_filename, to_print=None):
    eps = []
    rs = []
    with open(csv_filename, 'r') as f:
        for line in f:
            ep, reward = line.strip().split(',')
            eps.append(float(ep))
            rs.append(float(reward))
    l = [len(eps)]
    for i in l:
        if i >= 100:
            successes = np.sum(np.array(rs[max(0, i-100):i]) > 200)
            std = np.std(rs[max(0, i-100):i])
        else:
            successes = np.sum(np.array(rs[:i]) > 200)
            std = np.std(rs[:i])
        if i >= 200:
            mean = np.mean(rs[max(0, i-200):i])
        else:
            mean = np.mean(rs[:i]) if i > 0 else 0
        if successes >= 95 and mean >= 180:
            # print(f"{to_print} - for plotting code: {i}")
            return True, mean, std
    return False


for file in os.listdir('saved_models'):
    if file.startswith("iteration_") and file.endswith(".pth"):
        parts = file.split('_')
        if len(parts) >= 4:
            n = parts[1]
            conv = parts[3].split('.')[0]

    csv_filename = f"consistency_analysis/plottable_lunar_lander_final_{str(n)}.csv"
    data[str(n)] = {
        "iteration number": n,
        "csv filename": csv_filename,
        "model filename": file,
        "convergence episode": conv,
        "episodes": get_data(csv_filename, "Episode"),
        "rewards": get_data(csv_filename, "Reward"),
    }
    csv_filename = f"consistency_analysis/lunar_lander_final_{str(n)}.csv"
    mean, std = 0, 0
    to_print = f"iteration:{n}, conv for training code: {conv}"
    converged, mean, std = convergence_check(csv_filename, to_print=to_print)
    data[str(n)]["mean"] = mean
    data[str(n)]["std"] = std
    if not converged:
        print(f"error: iteration {n} has not been trained to convergence")




means = []
stds = []
convs = []

for n in data:
    means.append(data[str(n)]["mean"])
    stds.append(data[str(n)]["std"])
    convs.append(int(data[str(n)]["convergence episode"]))

print(means)
print(stds)
print(convs)

# scatter = plt.scatter(means, convs, c=stds, cmap='viridis', s=100)
# plt.xlabel('Mean Reward')
# plt.ylabel('Convergence Episode')
# # plt.title('Mean Reward vs Convergence Episode (colored by Std)')
# cbar = plt.colorbar(scatter)
# cbar.set_label('Reward Std')
# plt.show()


# stds = []
# means = []
# convs = []

for it in data:
    stds.append(data[it]["std"])
    means.append(data[it]["mean"])
    convs.append(int(data[it]["convergence episode"]))

# Normalize all arrays to [0, 1] for fair comparison
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else arr

stds_norm = normalize(stds)
means_norm = normalize(means)
convs_norm = normalize(convs)

fig, ax_conv = plt.subplots(figsize=(10, 6))

# Prepare bins and density curves for normalized data
conv_bins = np.linspace(0, 1, 100)
mean_bins = np.linspace(0, 1, 100)
std_bins = np.linspace(0, 1, 100)

# KDE for each normalized variable
conv_kde = gaussian_kde(convs_norm)
mean_kde = gaussian_kde(means_norm)
std_kde = gaussian_kde(stds_norm)

ax_mean = ax_conv.twiny()
ax_std = ax_conv.twiny()

# Move all x-axes to the bottom
ax_conv.xaxis.set_ticks_position('bottom')
ax_conv.xaxis.set_label_position('bottom')
ax_mean.xaxis.set_ticks_position('bottom')
ax_mean.xaxis.set_label_position('bottom')
ax_std.xaxis.set_ticks_position('bottom')
ax_std.xaxis.set_label_position('bottom')

# Offset the axes vertically so all are visible
ax_mean.spines['bottom'].set_position(('outward', 40))
ax_std.spines['bottom'].set_position(('outward', 80))

# Hide the top spines
ax_conv.spines['top'].set_visible(False)
ax_mean.spines['top'].set_visible(False)
ax_std.spines['top'].set_visible(False)

# Plot KDE curves
ax_conv.plot(conv_bins, conv_kde(conv_bins), color='orange', label='Convergence Episode')
ax_mean.plot(mean_bins, mean_kde(mean_bins), color='green', label='Mean Reward')
ax_std.plot(std_bins, std_kde(std_bins), color='blue', label='Reward Std')

# Fill the area under each curve with translucent color
ax_conv.fill_between(conv_bins, conv_kde(conv_bins), color='orange', alpha=0.2)
ax_mean.fill_between(mean_bins, mean_kde(mean_bins), color='green', alpha=0.2)
ax_std.fill_between(std_bins, std_kde(std_bins), color='blue', alpha=0.2)

# Mark the average value of each curve
mean_conv_norm = (np.mean(convs) - np.min(convs)) / (np.max(convs) - np.min(convs)) if np.max(convs) > np.min(convs) else 0
mean_mean_norm = (np.mean(means) - np.min(means)) / (np.max(means) - np.min(means)) if np.max(means) > np.min(means) else 0
mean_std_norm = (np.mean(stds) - np.min(stds)) / (np.max(stds) - np.min(stds)) if np.max(stds) > np.min(stds) else 0

ax_conv.axvline(mean_conv_norm, color='orange', linestyle='--', linewidth=2, label='Avg Conv')
ax_mean.axvline(mean_mean_norm, color='green', linestyle='--', linewidth=2, label='Avg Mean')
ax_std.axvline(mean_std_norm, color='blue', linestyle='--', linewidth=2, label='Avg Std')

# Set labels
ax_conv.set_xlabel('Convergence Episode')
ax_mean.set_xlabel('Mean Reward at Convergence')
ax_std.set_xlabel('Reward Standard Deviation after Convergence')
ax_conv.set_ylabel('Density')

# Set tick colors for clarity
ax_conv.xaxis.label.set_color('orange')
ax_mean.xaxis.label.set_color('green')
ax_std.xaxis.label.set_color('blue')

# Set custom ticks for each axis to show actual values, including the average
def set_actual_ticks(ax, orig_arr, norm_arr, bins, avg_norm, avg_val):
    # Choose a few representative ticks (min, median, max, avg)
    ticks_norm = [0, 0.5, 1, avg_norm]
    ticks_actual = [
        int(np.min(orig_arr)),
        int(np.median(orig_arr)),
        int(np.max(orig_arr)),
        avg_val
    ]
    # Remove duplicates and sort by normalized value
    ticks = sorted(zip(ticks_norm, ticks_actual), key=lambda x: x[0])
    ticks_norm_sorted, ticks_actual_sorted = zip(*ticks)
    ax.set_xticks(ticks_norm_sorted)
    ax.set_xticklabels([f"{v:.2f}" for v in ticks_actual_sorted])

set_actual_ticks(ax_conv, convs, convs_norm, conv_bins, mean_conv_norm, np.mean(convs))
set_actual_ticks(ax_mean, means, means_norm, mean_bins, mean_mean_norm, np.mean(means))
set_actual_ticks(ax_std, stds, stds_norm, std_bins, mean_std_norm, np.mean(stds))

# Add vertical lines at the borders of each bin for each axis
for b in conv_bins:
    ax_conv.axvline(b, color='orange', linestyle=':', alpha=0.1, linewidth=0.7)
for b in mean_bins:
    ax_mean.axvline(b, color='green', linestyle=':', alpha=0.1, linewidth=0.7)
for b in std_bins:
    ax_std.axvline(b, color='blue', linestyle=':', alpha=0.1, linewidth=0.7)



plt.tight_layout()
plt.show()
