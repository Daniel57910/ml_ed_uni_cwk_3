import pandas as pd
import matplotlib.pyplot as plt
RESULTS_FILES_DIR = 'results_2022_03_06_10'



df_train, df_val = (
    pd.read_csv(RESULTS_FILES_DIR + "/training_10_35.csv"),
    pd.read_csv(RESULTS_FILES_DIR + "/validation_10_35.csv"),
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

print(df_combined[['accuracy_train', 'accuracy_val', 'losses_train', 'losses_val']])

fig, axs = plt.subplots(1, 2)
plotting_meta = [
    {
        "cols": ['losses_train', 'losses_val'],
        "title": "BCE on training and validation dataset"
    },
    {
        "cols":["accuracy_train", "accuracy_val"],
        "title": "Accuracy score on training and validation dataset"
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