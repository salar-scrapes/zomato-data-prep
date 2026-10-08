import pandas as pd

df = pd.read_csv('zomato.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)
df = df.rename(columns={'approx_costfor_two_people': 'approx_cost'})

print(f"rows: {len(df):,} | unique url: {df['url'].nunique():,} | exact duplicate rows: {df.duplicated().sum():,}")

key = ['name', 'address']
t = df.copy()
t['rate_n'] = pd.to_numeric(t['rate'].astype(str).str.replace('/5', '', regex=False).str.strip(), errors='coerce')
t['cost_n'] = pd.to_numeric(t['approx_cost'].astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')
t['phone_n'] = t['phone'].str.replace(r'\D', '', regex=True).where(lambda s: s != '')

g = t.groupby(key)
print(f"outlets (name + address): {g.ngroups:,}")
for label, col in [('votes', 'votes'), ('rate', 'rate_n'), ('approx_cost', 'cost_n'),
                   ('location', 'location'), ('phone (digits only)', 'phone_n')]:
    print(f"differing {label}: {g[col].nunique().gt(1).sum():,}")
n_any = g[['rate_n', 'votes', 'cost_n']].nunique().gt(1).any(axis=1).sum()
print(f"differing rate/votes/approx_cost (any): {n_any:,}")