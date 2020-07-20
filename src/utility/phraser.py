'''
Detect phrases in text.

a) POS filtering
b) phrase detection

TODO:
- retrun a list of dataframes with POS and all that.
'''

import pandas as pd
import ndjson

from gensim import models, corpora
import text_to_x as ttx


def train(texts, lang='da', tokentype='lemma', out_path=None):
    '''Run gensim phrase detection, remove empty, keep dates.
    Returns lists of tokens.
    
    Parameters
    ----------
    texts : list
        Assuming that texts have _already_ been preprocessed
        using text_to_x 
        (i.e. tokenization, lemmatization & feature selection)

    lang : str
        Two-character ISO code of a desired language.

    tokentype : str
        Either "token" or "lemma".

    out_path : str (optional)
        path to a directory, where results will be saved
        (in a child directory).
    '''
    # convert to a nice format
    # keep only "meaningful" POS
    # (i.e. noun, propnoun, adj, verb, adverb)
    ttt_wrap = ttx.TextToTokensWrapper(texts, lang=lang)
    mega_ttx = ttx.TextToTopic(ttt_wrap, tokentype=tokentype)

    # initialize phrase detection
    phrases = models.Phrases(mega_ttx.tokenlists, delimiter=b" ")
    # find phrases
    phraser = models.phrases.Phraser(phrases)
    # extract texts with phrases detected
    phrase_list = [phraser[tl] for tl in mega_ttx.tokenlists]

    # missing any data?
    assert len(mega_ttx.tokenlists) == len(phrase_list)

    # put together dates and tweets
    phrase_doc = []
    for (i, tweet) in enumerate(phrase_list):
        d = dict()
        d['text'] = tweet
        phrase_doc.append(d)

    # remove empty tweets
    phrase_doc = [doc for doc in phrase_doc if doc['text']]
    texts = [doc['text'] for doc in phrase_doc]

    # if saving enabled
    if out_path:
        # check if file extension is specified
        if out_path.endswith('.ndjson'):
            pass
        # add it automatically if not
        else:
            print("Adding file extension (.ndjson)")
            out_path = os.path.join(out_path, '.ndjson')

        # export it
        with open(out_path, 'w') as f:
            ndjson.dump(phrase_doc, f)

    return texts
