import pandas as pd

# # iteration one
# # Load data from .csv files
# traindf = pd.read_csv('UNSW_NB15_training-set.csv')
# testdf = pd.read_csv('UNSW_NB15_testing-set.csv')

# # take the first 18 rows that have attack_cat = normal and the first 6 rows that have attack_cat = DoS
# normalcut = traindf[traindf['attack_cat'] == 'Normal'].head(18)
# doscut = traindf[traindf['attack_cat'] == 'DoS'].head(6)

# # put together
# traindf = pd.concat([normalcut, doscut])

# # create col and number each of the rows in order
# traindf['connectionnumber'] = range(1, len(traindf) + 1)

# print(traindf)

# # save to new csv
# traindf.to_csv('customdata.csv', index=False)

# iteration 2
# Load data from .csv files
df = pd.read_csv('customdata.csv')

# drop attack_cat column
df = df.drop(columns=['attack_cat', ])
