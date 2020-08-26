'''
Operations with query terms given to us by the researchers.
'''
import numpy as np
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors

import text_to_x as ttx


def import_query(ordlist_path, lang, col='word', collapse=True):
    '''
    Get query terms in & preprocess.
    '''
    # query
    query = (pd.read_csv(ordlist_path)
             .dropna(subset=[col]))
    query_words = [row for row in query.loc[:,col]]

    # preprocess query terms
    ttt_da = ttx.TextToTokens(lang=lang)
    ttt_da.texts_to_tokens(texts=query_words)
    query_dfs = ttt_da.get_token_dfs()
    
    # convert to a list of unigrams
    query_list = []
    for df in query_dfs:
        one_phrase = [word for word in df['token']]
        query_list.append(one_phrase)
        
    if collapse:
        # collapse back together
        query_list = [' '.join(ngram)
                      for ngram in query_list]
    
    return query_list


def sim_report(model, query, topn=10, include_query=True, out_df=False):
    '''
    Dataframe of words similar to query + their frequency
    '''
    # frequency of query
    query_count = model.vocab[query].count
    # get similarities
    sim_list = model.similar_by_word(query)
    # get frequency for each term
    sim_list_freq = [pair
                     # add frequency of term (as a tuple)
                     +(model.vocab[pair[0]].count,)
                     # iterate over term-simiarity pairs
                     for pair in sim_list]

    if include_query:
        # add info about query to the start
        # (term, similarty of 1, frequency of query)
        sim_list_freq.insert(0, (query, 1, query_count))
        
    if out_df:
        sim_list_freq = pd.DataFrame(sim_list_freq,
                                     columns=['related', 'similarity', 'count'])
        sim_list_freq.insert(0, 'term', query)
    
    return sim_list_freq


def get_related(model, terms, topn=10, cutoff=500):
    '''
    From a gensim model, extract most related words to query.

    Parameters
    ----------
    model : gensim.KeyedVectors | Word2Vec
        Trained gensim model with "vocab" & "similar_by_word" methods.
        
    terms : str|list|iterable
        Words to process. List or other iterable.
        
    topn : int
        How many most similar words to extract?
        Default 10.

    cutoff : int
        Keep only related words appearing this many times.
        Default 500.
    '''
    all_related = pd.DataFrame([])
    for term in terms:
        try:
            one = sim_report(model, term, out_df=True)
            all_related = all_related.append(one)
        except KeyError:
            one = pd.DataFrame({'term': [term],
                                'related': np.nan,
                                'similarity': np.nan,
                                'count': np.nan})
            all_related = all_related.append(one)
            
    hf_related = all_related.query('count >= @cutoff')
    hf_related['similarity'] = round(hf_related['similarity'], 2)
    
    return hf_related
