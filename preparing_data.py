import numpy as np
import pandas as pd
from scipy.sparse import data 

def prepare_data(file_name:str):
    train_df = pd.read_csv('data/' + file_name + '.csv')
    clients_df = pd.read_csv('data/cleaned_cliens.csv')

    clients_df = clients_df.merge(train_df.client_id, how='inner')

    train_purch_df = pd.read_csv('data/' +  file_name + '_purch.csv')
    train_purch_df['transaction_datetime'] = pd.to_datetime(train_purch_df['transaction_datetime'], errors='coerce')
    
    last_purch_ts = train_purch_df.groupby('client_id', as_index=False).transaction_datetime.max().rename(columns={'transaction_datetime': 'last_ts'})
    last_purch_ts['last_ts'] = last_purch_ts.last_ts.values.view('int64')/(1e9)

    train_purch_df['ts'] = train_purch_df.transaction_datetime.values.view('int64')/(1e9)
    
    train_purch_df = train_purch_df.merge(last_purch_ts, on='client_id', how='outer')
    
    train_purch_without_last_purch = train_purch_df.loc[train_purch_df.ts < train_purch_df.last_ts]
    pred_last_purch_ts = train_purch_without_last_purch.groupby('client_id', as_index=False).ts.max()
    pred_last_purch_ts = pred_last_purch_ts.rename(columns={'ts': 'pred_last_ts'})
    purch_ts = last_purch_ts.merge(pred_last_purch_ts, how='outer')
    purch_ts.pred_last_ts = purch_ts.pred_last_ts.fillna(purch_ts.last_ts)
    clients_df = clients_df.merge(purch_ts, how='inner')
    
    clients_df['days_own_bc'] = (clients_df.pred_last_ts - clients_df.first_reedem_ts) / (24*60*60)
    clients_df['days_from_last_purch'] = (clients_df.last_ts - clients_df.pred_last_ts) / (24*60*60)
    
    prods_data = pd.read_csv('data/products.csv')
    count_of_prds = train_purch_df.groupby('product_id', as_index=False).client_id.count()
    count_of_prds = count_of_prds.merge(prods_data.loc[:,['product_id', 'is_own_trademark', 'is_alcohol']], on='product_id', how='inner')
    count_of_prds = count_of_prds.rename(columns={'client_id':'purchases'})
    train_purch_df = train_purch_df.merge(count_of_prds.loc[:, ['product_id', 'is_own_trademark', 'is_alcohol']], on='product_id', how='outer')

    train_purch_df = train_purch_df.merge(clients_df.loc[:, ['client_id', 'first_reedem_ts']])    
    count_purch_before_fr = train_purch_df.loc[train_purch_df.ts < train_purch_df.first_reedem_ts].groupby('client_id', as_index=False).product_id.count()
    count_purch_before_fr = count_purch_before_fr.rename(columns={'product_id': 'count_purch_before_fr'})
    
    count_purch_after_fr = train_purch_df.loc[train_purch_df.ts >= train_purch_df.first_reedem_ts].groupby('client_id', as_index=False).product_id.count()
    count_purch_after_fr = count_purch_after_fr.rename(columns={'product_id': 'count_purch_after_fr'})

    clients_df = clients_df.merge(count_purch_before_fr, how='outer')
    clients_df = clients_df.merge(count_purch_after_fr, how='outer')
    clients_df = clients_df.fillna(0)

    transactions_data = train_purch_df.groupby('transaction_id', as_index=False).agg({'client_id':'max', 'purchase_sum':'mean', 'is_own_trademark':'max', 'is_alcohol':'max', 'express_points_spent':'mean'})
    client_info = transactions_data.groupby('client_id', as_index=False).agg({'is_own_trademark':'sum', 'is_alcohol':'sum', 'transaction_id':'count', 'purchase_sum':'sum', 'express_points_spent':'sum'})
    client_info = client_info.rename(columns={'is_own_trademark':'count_own_trm', 'is_alcohol':'count_alc', 'transaction_id':'purchs_count', 'express_points_spent':'express_points_spent_sum'})
    client_info['express_points_spent_mean'] = client_info.express_points_spent_sum / client_info.purchs_count
    client_info['purchase_sum_mean'] = client_info.purchase_sum / client_info.purchs_count
    clients_df = clients_df.merge(client_info, on='client_id', how='inner')
    
    train_df = train_df.merge(clients_df.drop(columns=['first_issue_date', 'first_redeem_date']), on='client_id', how='inner')
    train_df.to_csv('Prepared_data_' + file_name + '.csv', index=False)





