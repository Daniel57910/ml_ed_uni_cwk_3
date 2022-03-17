import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', help='results_directory')
args = parser.parse_args()
results_directory = args.d
results_files = os.listdir(results_directory)
training_csv = next(f for f in results_files if 'training' in f)
validation_csv = next(f for f in results_files if 'validation' in f)

if 'nus' in training_csv and 'nus' in validation_csv:
    project = 'Nus'
elif 'coco' in training_csv and 'coco' in validation_csv:
    project = 'Coco'
else:
    raise Exception("Ensure results either focusing on Nus or Coco")

df_train_path, df_val_path = os.path.join(results_directory, training_csv), os.path.join(results_directory, validation_csv)
print(df_train_path, df_val_path)

df_train, df_val = (
    pd.read_csv(df_train_path),
    pd.read_csv(df_val_path),
)

df_train, df_val = (
    df_train.groupby('epoch').mean().reset_index(),
    df_val.groupby('epoch').mean().reset_index()
)

df_train.columns = [col + "_train"  if col != 'epoch' else col for col in df_train.columns]
df_val.columns = [col + "_val" if col != 'epoch' else col for col in df_val.columns]

df_combined = pd.merge(
    df_train,
    df_val,
    on='epoch'
)

print(df_combined.columns)
print(df_combined[['mAP_train', 'mAP_val', 'losses_train', 'losses_val']])

fig, axs = plt.subplots(1, 2)
plotting_meta = [
    {
        "cols": ['losses_train', 'losses_val'],
        "title": f"BCE on {project} training and validation dataset"
    },
    {
        "cols":["mAP_train", "mAP_val"],
        "title": f"Mean Average Precision on {project} training and validation dataset"
    }
]
for ax, meta in zip(axs.flatten(), plotting_meta):
    subset = df_combined[meta['cols']]
    subset.plot(
        ax=ax,
        title=meta['title']
    )

    ax.set_xlabel("Epoch")

plt.show()