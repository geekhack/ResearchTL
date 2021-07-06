# importing panda library
import pandas as pd
import scipy.stats as stats1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm,kurtosis,entropy
from scipy.stats import skew

all_insects_df = pd.read_csv("../csv/reduced_insects_data.csv")


#get the value counts
print(all_insects_df['label'].value_counts())
#check the skewness of the data
print("Skewness:", skew(all_insects_df['class_id']))

#check the figure using a histogram
plt.figure()
sns.distplot(all_insects_df['class_id'])
plt.show()

print("Kurtosis:",all_insects_df['class_id'].kurt())
x = np.linspace(-5, 5, 100)

ax = plt.subplot()
distnames = ['laplace', 'norm', 'uniform']

for distname in distnames:
    if distname == 'uniform':
        dist = getattr(stats1, distname)(loc=-2, scale=4)
    else:
        dist = getattr(stats1, distname)
    data = all_insects_df['class_id']
    kur = kurtosis(data, fisher=True)
    y = dist.pdf(x)
    ax.plot(x, y, label="{}, {}".format(distname, round(kur, 3)))
    ax.legend()
    plt.show()

#calculate the entropy of the flowers dataset
print("Entropy:",entropy(all_insects_df['class_id'].value_counts()))



