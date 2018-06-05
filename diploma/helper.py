from random import randint
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def scale(x):
    return np.log(x + 1)
   

class customScaler():
    def __init__(self, feature_range=(0,100)):
        self.feature_range = feature_range
    
    def fit(self, x):
        scaled_data = scale(np.array(x, dtype=np.float))
        self.scaler = MinMaxScaler(feature_range=self.feature_range)
        self.scaler.fit(scaled_data)

    def transform(self, data):
        scaled_data = scale(np.array(data, dtype=np.float))
        transformed = self.scaler.transform(scaled_data).astype(int)
        return np.array(transformed, dtype=np.int64)
    
    

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
        scaler.fit(np.append(data.values, np.array([[0, 0]]), axis=0))
        groupped_data = pd.DataFrame(scaler.transform(data), columns=['number of flows', 'mean(byte count)'])
        groupped_data['source computer'] = data_reset['source computer']
    else:
        groupped_data = pd.DataFrame(data.values, columns=['number of flows', 'mean(byte count)'])
        groupped_data['source computer'] = data_reset['source computer']
    return groupped_data, hosts

def clear_df(df, hosts):
    return df[df['source computer'].isin(hosts)]

def group_scale_data(df, size_of_bin_seconds=60, doScale=False, scaler='log', addZeros=True, hosts=None, verbose=0):
    """
    :param size_of_bin_seconds: the time period of each bin,
                assumes the dataframe has a column names 'source computer' and a name 'byte count'
    :param addZeros: add values (0, 0) where no data has been received for this bucket
    :return: a dictionary containing for each host the features, the hosts 
    """
                     
    # only log scale offered 
    assert scaler == 'log'
    if doScale and scaler == 'log':
        scaler = customScaler()

    if hosts is None:
        hosts = np.array(list(set(df['source computer'].values)))
    else:
        df = clear_df(df, hosts)

    bins = np.arange(df.index.min(), df.index.max() + size_of_bin_seconds + 1, size_of_bin_seconds)

    groups = df[['byte count','source computer']].groupby([np.digitize(df.index, bins),'source computer'])

    data = groups.count()
    data.columns = ['number of flows']
    data['mean(byte count)'] = groups.mean().values

    data_reset = data.reset_index()

    if verbose > 0:
        print('A total of', len(bins) - 1, 'time epochs have been encountered')
    
    len_hosts = len(hosts)
    intervals = int(len_hosts / 10)
    i = 0
    
    if addZeros:
        add_new = []
        for host in hosts:
            if verbose > 0 and i % intervals == 0:
                print('Done with', i, 'hosts out of', len_hosts)
            i += 1

            for bin_i in range(1,len(bins)):
                if (bin_i, host) not in data.index:
                    new_row = [bin_i, host, 0.0, 0.0]
                    add_new.append(new_row)
                    
        
        data_reset = data_reset.append(pd.DataFrame(add_new, columns=data_reset.columns), ignore_index=True )

    if verbose > 0:
        print('Scaling...')
    if doScale:
        scaler.fit(np.append(data_reset.values[:,2:], np.array([[0, 0]]), axis=0))
        groupped_data = pd.DataFrame(scaler.transform(data_reset.values[:,2:]), columns=['number of flows', 'mean(byte count)'])
        groupped_data['epoch'] = data_reset['level_0']
        groupped_data['source computer'] = data_reset['source computer']
    else:
        groupped_data = pd.DataFrame(data_reset.values[:,2:], columns=['number of flows', 'mean(byte count)'])
        groupped_data['epoch'] = data_reset['level_0']
        groupped_data['source computer'] = data_reset['source computer']

    # set parameters for next acquisition of data
    parameters = {}
    parameters['scaler'] = scaler
    parameters['doScale'] = doScale
    parameters['size_of_bin_seconds'] = size_of_bin_seconds
    parameters['hosts'] = hosts
    parameters['addZeros'] = addZeros

    groupped_data = groupped_data.sample(frac=1)
    
    return groupped_data.sort_values(by=['epoch']), hosts, parameters

def group_scale_data_batch(df, parameters, setHosts=False, verbose=0):
    """
    :param setHosts: get the new hosts if True else get the hosts we are interested in from the parameters
    """
    
    scaler = parameters['scaler']
    doScale = parameters['doScale']
    size_of_bin_seconds = parameters['size_of_bin_seconds']  
    addZeros = parameters['addZeros'] 
    
    if setHosts:
        hosts = np.array(list(set(df['source computer'].values)))
    else:
        hosts = parameters['hosts']
        df = clear_df(df, hosts)

    bins = np.arange(df.index.min(), df.index.max() + size_of_bin_seconds + 1, size_of_bin_seconds)

    groups = df[['byte count','source computer']].groupby([np.digitize(df.index, bins),'source computer'])

    data = groups.count()
    data.columns = ['number of flows']
    data['mean(byte count)'] = groups.mean().values

    data_reset = data.reset_index()
         
    len_hosts = len(hosts)
    intervals = int(len_hosts / 10)
    i = 0       
    
    if addZeros:
        add_new = []
        for host in hosts:
            if verbose > 0 and i % intervals == 0:
                print('Done with', i, 'hosts out of', len_hosts)
            i += 1

            for bin_i in range(1,len(bins)):
                if (bin_i, host) not in data.index:
                    new_row = [bin_i, host, 0.0, 0.0]
                    add_new.append(new_row)
                    
        
        data_reset = data_reset.append(pd.DataFrame(add_new, columns=data_reset.columns), ignore_index=True )

    if doScale:
        groupped_data = pd.DataFrame(scaler.transform(data_reset.values[:,2:]), columns=['number of flows', 'mean(byte count)'])
        groupped_data['epoch'] = data_reset['level_0']
        groupped_data['source computer'] = data_reset['source computer']
    else:
        groupped_data = pd.DataFrame(data_reset.values[:,2:], columns=['number of flows', 'mean(byte count)'])
        groupped_data['epoch'] = data_reset['level_0']
        groupped_data['source computer'] = data_reset['source computer']

    groupped_data = groupped_data.sample(frac=1)
    
    return groupped_data.sort_values(by=['epoch']), hosts