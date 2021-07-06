import pandas as pd

train_df = pd.read_csv("../csv/skew_cubs_data.csv")
# get the value counts

#for i in range(1,197):
n = []
for c in range(1,11):
    m = []
    for index, row in train_df.iterrows():
        if row['class_id'] == c:
            m.append((row['file'], row['label'], row['class_id']))

    n.append(m)

x = []

for i in range(len(n)):
    # for j in range(len(n[i][:round(len(n[i]) / 6)])):
    #     x.append(n[i][j])
        for j in range(len(n[i][:round(len(n[i]))])):
            x.append(n[i][j])
df = pd.DataFrame(x)
df.columns = ['file', 'label', 'class_id']
df.to_csv('../csv/reduced_10_cubs_data.csv', index=False)