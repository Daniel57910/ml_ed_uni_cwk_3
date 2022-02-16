import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_results_2022_02_15-19:34.csv")
df = df.groupby('epoch').mean()
print(df.columns)
df['micro/recall'].plot(
    kind='line',
    title='Recall Per Epoch On Training Data',
    y='Binary Cross Entropy')
plt.show()