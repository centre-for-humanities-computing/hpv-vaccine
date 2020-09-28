'''Downsample a document-topic matrix to daily time bins.

TODO
----
- parse_timebin_size()
    - implement in import_normalize()
- prediction method
    - concat documents in time bins
    - predict a topic distribution for
      that big daily document.
'''
import pandas as pd
import ndjson


def parse_timebin_size(timebin_size):
    '''
    Define temporal size of the time bin.
    
    Example
    -------
    prase_timebin_size('day') -> dates.groupby(day) etc.
    '''
    return None


def import_normalize(doctop_path, train_data_path, meta_data_path, datetime_col='time'):
    '''
    Import & normalize a document-topic matrix.

    So far, only the averaging method is implemented!
    '''
    # DOCTOP
    with open(doctop_path) as f:
        doctop = ndjson.load(f)
    # normalize
    norm_all = [[value / sum(doc) for value in doc]
                for doc in doctop]
    # to df
    norm_df = pd.DataFrame(norm_all)
    # test length
    assert norm_df.values.tolist() == norm_all


    # TRAIN DATA
    with open(train_data_path) as f:
        input_texts = ndjson.load(f)
    # iDs of documents used for training
    ids = [doc['id'] for doc in input_texts]


    # META DATA (dates)
    meta = pd.read_csv(meta_data_path, parse_dates=[datetime_col])
    # keep only docs used for training
    meta_trained = meta.iloc[ids, :]


    # AGGREGATE
    days = meta_trained['time'].dt.floor('d') # parse_timebin_size
    norm_df.index = days.index
    norm_df['days'] = days
    topic_col_names = [col for col in norm_df.columns if col!= 'days']
    # average
    avg_topic_df = norm_df.groupby('days')[topic_col_names].mean()
    doctop_avg = avg_topic_df.values.tolist()
    # normalize again
    doctop_avg = [[value / sum(doc) for value in doc]
                  for doc in doctop_avg]

    return doctop_avg
