import pandas as pd

#read the file with the class names
class_names = df1 = pd.read_csv("../utils/insect_classes.txt", skiprows=0, delim_whitespace=True, names=range(2))
# create csv file
class_names.columns = ['class_id', 'label']

class_names.to_csv('../csv/insect_classes.csv', index=None)

#read the insect training file
df_1 = pd.read_csv("../utils/insect_train.txt", skiprows=0, delim_whitespace=True, names=range(2))

# create csv file
df_1.columns = ['file', 'class_id']

df_1.to_csv('../csv/insect_train.csv', index=None)

#read the insect test file
df2 = pd.read_csv("../utils/insect_test.txt", skiprows=0, delim_whitespace=True, names=range(2))

# create csv file
df2.columns = ['file', 'class_id']

df2.to_csv('../csv/insect_test.csv', index=None)

#read the insect validation file
df3 = pd.read_csv("../utils/insect_val.txt", skiprows=0, delim_whitespace=True, names=range(2))

# create csv file
df3.columns = ['file', 'class_id']

df3.to_csv('../csv/insect_val.csv', index=None)



#read from csvs
train_df = pd.read_csv("../csv/insect_train.csv")
test_df = pd.read_csv("../csv/insect_test.csv")
val_df = pd.read_csv("../csv/insect_val.csv")
class_names_df = pd.read_csv("../csv/insect_classes.csv")

train_df_classes = pd.merge(train_df, class_names_df, on=['class_id'], how='inner')
test_df_classes = pd.merge(test_df, class_names_df, on=['class_id'], how='inner')
val_df_classes = pd.merge(val_df, class_names_df, on=['class_id'], how='inner')

#get the value counts
train_class_labels = train_df['class_id'].unique().tolist()
#print(train_class_labels)
test_class_labels = test_df['class_id'].unique().tolist()
#print(test_class_labels)
val_class_labels = val_df['class_id'].unique().tolist()
#print(val_class_labels)

#combine the three into one dataset to check for the data biasness
train_test=train_df_classes.append(test_df_classes)
train_test_val = train_test.append(val_df_classes)

train_test_val.to_csv('../csv/merged_insects.csv', index = None)
