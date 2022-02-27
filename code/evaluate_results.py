import pandas as pd
import matplotlib.pyplot as plt
RESULTS_FILES_DIR = 'results_files'


df_train, df_val = (
    pd.read_csv(RESULTS_FILES_DIR + "/training_results_2022_02_16-15:03.csv"),
    pd.read_csv(RESULTS_FILES_DIR + "/validation_results_2022_02_16-15:03.csv")
)

df_train, df_val = (
    df_train.groupby('epoch').mean().reset_index(),
    df_val.groupby('epoch').mean().reset_index()
)

df_train.columns = [col + "_train"  if col != 'epoch' else col for col in df_train.columns]
df_val.columns = [col + "_val" if col != 'epoch' else col for col in df_val.columns]
print(df_train.columns)

df_combined = pd.merge(
    df_train,
    df_val,
    on='epoch'
)

print(df_combined.columns)
fig, axs = plt.subplots(2, 2)
plotting_meta = [
    {
        "cols": ['losses_train', 'losses_val'],
        "title": "BCE on training and validation dataset"
    },
    {
        "cols": ["micro/precision_train", "micro/precision_val"],
        "title": "Precision score on training and validation dataset"
    },
    {
        "cols": ["micro/recall_train", "micro/recall_val"],
        "title": "Recall score on training and validation dataset"
    },
    {
        "cols": ["micro/f1_train", "micro/f1_val"],
        "title": "F1 score on training and validation dataset"
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