from random import randint
import numpy as np
import pandas as pd

def get_random_initialize_lamdas(data, number_of_mixtures=4):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    
    dims = len(mins)
    lambdas = [[] for _ in range(number_of_mixtures)]
    for i in range(dims):
        for j in range(number_of_mixtures):
            lambdas[j].append(randint(int(mins[i]), int(maxs[i])))
            
    return np.vstack(lambdas)

# TODO remove this
def get_data_by_dataframe(df, size_of_bin_seconds=50, doScale=True, scaler=None):
    """
    :param size_of_bin_seconds: the time period of each bin,
                assumes the dataframe has a column names 'source computer' and a name 'byte count'
    :return: a dictionary containing for each host the features, the hosts 
    """
    hosts = np.array(list(set(df['source computer'].values)))
    
    bins = np.arange(df.index.min(), df.index.max() + size_of_bin_seconds + 1, size_of_bin_seconds)
    
    groups = df[['byte count','source computer']].groupby([np.digitize(df.index, bins),'source computer'])

    data = groups.count()
    data.columns = ['number of flows']
    data['mean(byte count)'] = groups.mean().values
    
    if doScale:
        scaler.fit(np.append(data.values, np.array([[0, 0]]), axis=0))
    
    data_by_host = {}
     
    for host in hosts:
        for i in range(len(bins) - 1):
            try:
                if doScale == True:
                    values = scaler.transform(np.array([data.loc[(i + 1, host)].values]))
                else:
                    values = np.array([data.loc[(i + 1, host)].values])
                    
                if i == 0:
                    data_by_host[host] = np.array(values)
                else:
                    data_by_host[host] = np.append(data_by_host[host], np.array(values), axis=0)
            except:
                pass
    return data_by_host, hosts

def group_data(df, size_of_bin_seconds=50, doScale=True, scaler=None):
    """
    :param size_of_bin_seconds: the time period of each bin,
                assumes the dataframe has a column names 'source computer' and a name 'byte count'
    :return: a dictionary containing for each host the features, the hosts 
    """
    hosts = np.array(list(set(df['source computer'].values)))
    
    bins = np.arange(df.index.min(), df.index.max() + size_of_bin_seconds + 1, size_of_bin_seconds)
    
    groups = df[['byte count','source computer']].groupby([np.digitize(df.index, bins),'source computer'])

    data = groups.count()
    data.columns = ['number of flows']
    data['mean(byte count)'] = groups.mean().values
    
    data_reset = data.reset_index()
    if doScale:
        scaler.fit(np.append(data.values, np.array([[1, 1]]), axis=0))
        groupped_data = pd.DataFrame(scaler.transform(data), columns=['number of flows', 'mean(byte count)'])
        groupped_data['source computer'] = data_reset['source computer']
    else:
        groupped_data = pd.DataFrame(data.values, columns=['number of flows', 'mean(byte count)'])
        groupped_data['source computer'] = data_reset['source computer']
    return groupped_data, hosts