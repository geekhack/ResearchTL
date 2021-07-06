import pandas as pd
import scipy.stats as stats1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm,kurtosis,entropy
from scipy.stats import skew


# then generate the datasets from the created csv

train_df = pd.read_csv("../csv/skew_sketches_data.csv")

#get the value counts
print(train_df['label'].value_counts())
#check the skewness of the data
print("Skewness:", skew(train_df['class_id']))

#check the figure using a histogram
plt.figure()
sns.distplot(train_df['class_id'])
plt.show()

print("Kurtosis:",train_df['class_id'].kurt())
x = np.linspace(-5, 5, 100)

ax = plt.subplot()
distnames = ['laplace', 'norm', 'uniform']

for distname in distnames:
    if distname == 'uniform':
        dist = getattr(stats1, distname)(loc=-2, scale=4)
    else:
        dist = getattr(stats1, distname)
    data = train_df['class_id']
    kur = kurtosis(data, fisher=True)
    y = dist.pdf(x)
    ax.plot(x, y, label="{}, {}".format(distname, round(kur, 3)))
    ax.legend()
    plt.show()

#calculate the entropy of the flowers dataset
print("Entropy:",entropy(train_df['class_id'].value_counts()))
