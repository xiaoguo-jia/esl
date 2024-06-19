import pandas as pd

# # iteration one
# # Load data from .csv files
# traindf = pd.read_csv('UNSW_NB15_training-set.csv')
# testdf = pd.read_csv('UNSW_NB15_testing-set.csv')

# # take the first 18 rows that have attack_cat = normal and the first 6 rows that have attack_cat = DoS
# normalcut = traindf[traindf['attack_cat'] == 'Normal'].head(42)
# doscut = traindf[traindf['attack_cat'] == 'DoS'].head(6)

# # put together
# traindf = pd.concat([normalcut, doscut])

# # create col and number each of the rows in order
# traindf['connectionnumber'] = range(1, len(traindf) + 1)

# print(traindf)

# # save to new csv
# traindf.to_csv('customdata.csv', index=False)





# # iteration 2
# # Load data from .csv files
# df = pd.read_csv('customdata.csv')

# # drop attack_cat, connectionnumber
# df = df.drop(columns=['proto', 'service', 'state', 'attack_cat', 'connectionnumber'])

# # make the id column match the index
# df['id'] = df.index

# # ones that are currently labelled for 1: 42, 44, 45, 46, 47
# # ones that should be labelled for 1: 28, 31, 8, 21, 23

# # # manually change values of 'id' column
# df.loc[42, 'id'] = 28
# df.loc[44, 'id'] = 8
# df.loc[45, 'id'] = 21
# df.loc[46, 'id'] = 23
# df.loc[47, 'id'] = 31

# df.loc[28, 'id'] = 42
# df.loc[31, 'id'] = 47
# df.loc[8, 'id'] = 44
# df.loc[21, 'id'] = 45
# df.loc[23, 'id'] = 46

# # short by decreasing 'id'
# df = df.sort_values(by='id', ascending=True)

# # save to new csv
# df.to_csv('customdata2.csv', index=False)

# df = pd.read_csv('customdata2.csv')
# print(df)




# # iteration 3
# df = pd.read_csv('customdata2.csv')

# # add columns 'source' and 'destination'
# listofsources = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7,  8, 9, 10, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15]
# listofdestinations= [1, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8, 9, 10, 11, 9, 10, 11, 12, 13, 14, 15, 13, 14, 15, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 4, 5, 6, 7,  8, 9, 10, 8, 9, 10, 11, 12, 13, 14]

# # add list of sources and destinations to the dataframe
# df['source'] = listofsources
# df['destination'] = listofdestinations

# # print rows what label is 1
# print(df[df['label'] == 1])

# # save to new csv
# df.to_csv('finaldata.csv', index=False)





# # test data
# # iteration one
# # Load data from .csv files
# traindf = pd.read_csv('UNSW_NB15_training-set.csv')
# testdf = pd.read_csv('UNSW_NB15_testing-set.csv')

# # take the first 22 rows that have attack_cat = normal and the first 2 rows that have attack_cat = DoS
# normalcut = testdf[testdf['attack_cat'] == 'Normal'].head(22)
# doscut = testdf[testdf['attack_cat'] == 'DoS'].head(2)

# # put together
# df = pd.concat([normalcut, doscut])

# # create col and number each of the rows in order

# print(df)

# # save to new csv
# df.to_csv('customtestdata.csv', index=False)




# # iteration 2
# # Load data from .csv files
# df = pd.read_csv('customtestdata.csv')

# # drop attack_cat, connectionnumber
# df = df.drop(columns=['proto', 'service', 'state', 'attack_cat'])

# # make the id column match the index
# df['id'] = df.index


# # ones that are currently labelled for 1: 22, 23
# # ones that should be labelled for 1: 9, 18

# # # manually change values of 'id' column
# df.loc[22, 'id'] = 9
# df.loc[9, 'id'] = 22
# df.loc[18, 'id'] = 23
# df.loc[23, 'id'] = 18


# # sort by decreasing 'id'
# df = df.sort_values(by='id', ascending=True)

# # save to new csv
# df.to_csv('customtestdata2.csv', index=False)

df = pd.read_csv('customtestdata2.csv')
print(df)






# iteration 3
df = pd.read_csv('customtestdata2.csv')

# add columns 'source' and 'destination'
listofsources =      [0, 1, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 7, 8]  
listofdestinations = [1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 7, 8, 0, 1, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7]  

# add list of sources and destinations to the dataframe
df['source'] = listofsources
df['destination'] = listofdestinations

# print rows what label is 1
print(df[df['label'] == 1])

# save to new csv
df.to_csv('finaltestdata.csv', index=False)