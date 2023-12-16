import os
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


root_dir = "./data/moos"
data_dir = "./data/moos/scenes"

scene_names = sorted(os.listdir(data_dir))
scene_dirs = [os.path.join(data_dir, scene_name) for scene_name in scene_names]

cats = ["chair", "bed", "table", "sofa"]
obj_counts_filepath = os.path.join(root_dir, "obj_counts.json")
obj_occ_rates_filepath = os.path.join(root_dir, "obj_occ_rates.json")
obj_counts = json.load(open(obj_counts_filepath))
obj_occ_rates = json.load(open(obj_occ_rates_filepath))

obj_occ_rates = json.load(open(obj_occ_rates_filepath))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 6))
axes = axes.flatten()
obj_idx_filtered = {}
for i, ax in enumerate(axes):
    ax.set_xlim(-0.05, 1.05)
    cat = cats[i]
    data = []
    for obj_id in obj_occ_rates[cat]:
        data.extend(obj_occ_rates[cat][obj_id])
    data = np.array(data)
    print(f"num of {cat} instances w/ some occlusion: {((data > 0.) & (data < 1.)).sum()}")
    print(f"num of {cat} instances w/o occlusion: {(data == 0.).sum()}")
    print(f"num of {cat} instances w/ full occlusion: {(data == 1.).sum()}")
    data_mask = (data > 0.0) & (data < 1.0)
    data_masked = data[data_mask]
    ax.hist(data_masked, bins=10, density=False)
    ax.set_title(cat)
    ax.set_xlabel('occlusion rate')
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.9,
                    hspace=0.4, wspace=0.3)
fig.savefig("obj_occlusion.png")


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 4))
obj_counts = json.load(open(obj_counts_filepath))
objects_sorted = {cat: defaultdict(lambda: []) for cat in cats}
for i, ax in enumerate(axes):
    cat = cats[i]
    objects_freq = sorted(list(obj_counts[cat].items()), key=lambda t: t[1], reverse=True)
    objects_sorted[cat] = objects_freq
    obj_ids, freqs = list(zip(*objects_freq))
    freqs = np.array(freqs, dtype=np.int32)
    freq_max = objects_freq[0][1]
    ax.hist(freqs, bins=range(1, freq_max+1), density=False)
    ax.set_title(cat)
    ax.set_xlabel('frequency of objects')
    print(f"num of unique {cat} objects: {len(obj_ids)}")
    for freq in range(freq_max, 0, -1):
        num_freq = (freqs == freq).sum()
        print(f"    num of objects with {freq} freqs: {num_freq}")
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.8,
                    hspace=0.4, wspace=0.3)
fig.savefig("obj_freq.png")
