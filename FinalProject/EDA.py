import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project = pd.read_csv("FinalProject2.csv")






project.to_csv("FinalProject3.csv", index=False, mode='a', header=True)
project.to_csv("C:/Users/cjy17/Desktop/project/FinalProject3.csv", index=False, mode='a', header=True)
x = project['Rating'].dropna()
y = project['Size'].dropna()
z = project['Installs'][project.Installs!=0].dropna()
p = project['Reviews'][project.Reviews!=0].dropna()
t = project['Type'].dropna()
price = project['Price']

p = sns.pairplot(pd.DataFrame(list(zip(x, y, z, np.log10(p),t,  price)),
                        columns=['Rating','Size', 'Installs', 'Reviews', 'Type', 'Price']), hue='Type', palette="Set2")

plt.show()

print(project.head())