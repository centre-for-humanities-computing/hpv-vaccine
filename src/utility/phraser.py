'''
Detect phrases in text.

a) POS filtering
b) phrase detection
'''

import pandas as pd
import ndjson

from gensim import models, corpora
import text_to_x as ttx


def train(texts, tokentype='lemma',
          allowed_pos=["NOUN", "ADJ", "VERB", "PROPN"],
          out_path=None):
    '''Run gensim phrase detection, remove empty, keep dates.
    Returns lists of tokens.
    
    Parameters
    ----------
    texts : list
        Assuming that texts have _already_ been preprocessed
        using text_to_x 
        (i.e. tokenization, lemmatization & feature selection)

    tokentype : str
        Either "token" or "lemma"

    allowed_pos : list
        uPOS that will be kept in texts.
        Use stanza tags.

    out_path : str (optional)
        path to a directory, where results will be saved
        (in a child directory).
    '''
    # convert to a nice format
    # keep only "meaningful" POS
    # (i.e. noun, propnoun, adj, verb, adverb)
    texts_filter = []
    for doc in texts:
        allowed_keys = [key for key, value in doc['upos'].items() if value in allowed_pos]
        texts_filter.append([word for key, word in doc[tokentype].items() if key in allowed_keys])

    # initialize phrase detection
    phrases = models.Phrases(texts_filter, delimiter=b" ")
    # find phrases
    phraser = models.phrases.Phraser(phrases)
    # extract texts with phrases detected
    phrase_list = [phraser[tl] for tl in texts_filter]

    # missing any data?
    assert len(texts_filter) == len(phrase_list)

    # put together IDs and documents
    phrase_doc = []
    for (i, tweet) in enumerate(phrase_list):
        d = dict()
        d['id'] = i
        d['text'] = tweet
        phrase_doc.append(d)

    # remove empty tweets
    phrase_doc = [doc for doc in phrase_doc if doc['text']]

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

    return phrase_doc
