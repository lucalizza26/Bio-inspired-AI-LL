import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

data = {}

layouts1 = []
layouts2 = []
layouts3 = []

for i in [64, 128, 256]:
    layouts1.append([i])
    for j in [64, 128, 256]:
        layouts2.append([i, j])
        for k in [64, 128, 256]:
            layouts3.append([i, j, k])

# for i in layouts1 + layouts2 + layouts3:
#     csv_filename = f"layout_analysis/lunar_lander_{'_'.join(map(str, i))}.csv"
#     new_csv_filename = f"layout_analysis/model_{'_'.join(map(str, i))}.csv"
#     with open(csv_filename, newline='') as infile, open(new_csv_filename, 'w', newline='') as outfile:
#         reader = csv.reader(infile)
#         writer = csv.writer(outfile)
#         writer.writerow(['Episode', 'Reward'])
#         for row in reader:
#             writer.writerow(row)

def get_data(csv_filename, column):
    values = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            values.append(float(row[column]))
    return values

for layout in layouts1 + layouts2 + layouts3:
    csv_filename = f"layout_analysis/model_{'_'.join(map(str, layout))}.csv"
    data[str(layout).strip('[]')] = {
        "depth": len(layout),
        "layout": layout,
        "csv_filename": csv_filename,
        "episodes": get_data(csv_filename, "Episode"),
        "rewards": get_data(csv_filename, "Reward"),
    }



def convergence(layout=str,):
    ep = data[layout]["episodes"]
    r = data[layout]["rewards"]
    for i in range(len(ep)):
        if i >= 100:
            successes = np.sum(np.array(r[max(0, i-100):i]) > 200)
            std = np.std(r[max(0, i-100):i])
        else:
            successes = np.sum(np.array(r[:i]) > 200)
            std = np.std(r[:i])
        if i >= 200:
            mean = np.mean(r[max(0, i-200):i])
        else:
            mean = np.mean(r[:i]) if i > 0 else 0
        if successes >= 95 and mean >= 180:
            print(f"layout {layout} converged at episode {ep[i]} - mean: {mean}, std: {std}")
            data[layout]["converged"] = True
            data[layout]["converged_episode"] = ep[i]
            data[layout]["mean"] = mean
            data[layout]["std"] = std
            return True
    return False


for layout in data:
    if not convergence(layout):
        print(f"Layout {layout} did not converge.")
        data[layout]["converged"] = False
        data[layout]["converged_episode"] = None
        data[layout]["mean"] = None
        data[layout]["std"] = None
    

# Prepare data for plotting
x = []
y = []
z = []
w = []

for layout in data:
    if data[layout]["converged"]:
        x.append(str(data[layout]["layout"]))
        y.append(data[layout]["converged_episode"])
        w.append(data[layout]["mean"])
        z.append(data[layout]["std"])
        # print(f"Layout {data[layout]['layout']} converged at episode {data[layout]['converged_episode']} - mean: {data[layout]['mean']}, std: {data[layout]['std']}")
    else:
        x.append(str(data[layout]["layout"]))
        y.append(1000)
        w.append(0)
        z.append(0)
        


# print(f"lrs: {x}")
# print(f"ep: {y}")
# print(f"mean: {w}")
# print(f"std: {z}")



# plt.scatter(x,y)
plt.xticks(rotation=90)

# Normalize z for colormap, skipping zeros
z_nonzero = [val for val in z if val != 0]
if z_nonzero:
    z_min, z_max = min(z_nonzero), max(z_nonzero)
else:
    z_min, z_max = 0, 1  # fallback to avoid error if all z are zero
norm = plt.Normalize(z_min, z_max)
cmap = plt.cm.viridis

# Normalize w for marker size, skipping zeros
w_nonzero = [val for val in w if val != 0]
if w_nonzero:
    w_min, w_max = min(w_nonzero), max(w_nonzero)
else:
    w_min, w_max = 0, 1  # fallback to avoid error if all w are zero

min_size = 50
max_size = 300

def get_marker_size(val):
    if val == 0 or w_max == w_min:
        return min_size
    return min_size + (max_size - min_size) * (val - w_min) / (w_max - w_min)

for i, val in enumerate(y):
    # Draw a light line from each bubble to the x-axis
    plt.plot([x[i], x[i]], [0, y[i]], color='lightgray', linewidth=1, zorder=1)
    if val == 1000:
        plt.scatter(x[i], y[i], marker='x', color='red', zorder=2)
    elif z[i] != 0:
        color = cmap(norm(z[i]))
        size = get_marker_size(w[i])
        plt.scatter(x[i], y[i], marker='o', color=color, s=size, zorder=2)

plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Standard Deviation')
plt.xlabel('Network Layout')
plt.ylabel('Convergence Episode')
plt.subplots_adjust(top=0.88, bottom=0.21, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
plt.ylim(480, 1020)
# Add legend for marker size (mean)

mean_legend_vals = [180, 210, 240]

handles = []
for mean_val in mean_legend_vals:
    handles.append(
        plt.scatter([], [], s=get_marker_size(mean_val), color='gray', alpha=0.6, label=f"Mean: {mean_val:.1f}")
    )

# Move the legend to the right outside the plot
plt.legend(handles=handles, title="Mean Reward (size)", loc='upper left', bbox_to_anchor=(1.14, 1))

plt.show()
