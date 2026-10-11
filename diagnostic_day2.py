import pandas as pd
df = pd.read_csv('zomato.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)
g = df.groupby(['name', 'address'])['reviews_list']
print('outlets with differing reviews_list:', g.nunique().gt(1).sum(), 'of', g.ngroups)
print(df['reviews_list'].str.len().describe())
print('mojibake share:', df['reviews_list'].str.contains('Ã', na=False).mean())
print(df['reviews_list'].dropna().iloc[1][:400])