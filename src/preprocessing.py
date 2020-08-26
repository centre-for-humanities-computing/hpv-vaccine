'''
Parallelized wrapper of text_to_x & Stanza.

TODO:
- finish documentation
- rename main function
- move this file somewhere
'''
import os
import argparse
import ndjson

import multiprocessing as mp
from functools import partial

import text_to_x as ttx


def __stanza(texts, lang):
    '''
    Use Stanza + Flair for Danish preprocessing.
    For other languages, use Stanza only.

    Parameters
    ----------
    texts : str|list
        Should be a string, a list or other iterable object.

    lang : str
        Two character ISO code of a desired language.
    '''
    if lang in {"da"}:
        ttt = ttx.TextToTokens(lang=lang,
                               tokenize="stanza",
                               lemmatize="stanza",
                               pos="stanza",
                               depparse="stanza",
                               ner="flair", silent=False)

    else:
        ttt = ttx.TextToTokens(lang=lang,
                               tokenize="stanza",
                               lemmatize="stanza",
                               pos="stanza",
                               depparse="stanza",
                               ner="stanza", silent=False)
    ttt.texts_to_tokens(texts)
    dfs = ttt.get_token_dfs()
    return dfs


def stanza_multicore(texts, lang, n_jobs_gpu):
    '''
    Run Stanza in multiprocessing.

    Parameters
    ----------
    texts : str|list
        Should be a string, a list or other iterable object.

    lang : str
        Two character ISO code of a desired language.

    n_jobs_gpu : int
        Number of workers to split the job between.
    '''
    pool_gpu = mp.Pool(n_jobs_gpu)
    length = len(texts)
    inc = round(length/4)
    l_texts = [texts[:inc],
               texts[inc:inc*2],
               texts[inc*2:inc*3],
               texts[inc*3:]]

    __stanza_ = partial(__stanza, lang=lang)

    dfs = pool_gpu.map(__stanza_, l_texts)
    pool_gpu.close()
    dfs = [ii for i in dfs for ii in i]  # unpack
    dfs_out = [df.to_dict() for df in dfs]

    return dfs_out


def _delete_many_equal_signs(dfs_out):
    '''
    Hotfix for weird strings crashing Stanza.

    - '========================'
    '''
    dfs_eq = [
        {'ID': doc['ID'],
         'text': doc['text'].translate(str.maketrans('', '', '\n\n========================'))
        }
        for doc in dfs_out
    ]
    
    return dfs_eq


def main():
    '''
    Run preprocessing
    '''
    # initialize argparser with a desc
    ap = argparse.ArgumentParser(
        description="Parallelized maximal preprocessing using stanza"
    )

    # input path
    ap.add_argument("-p", "--inpath",
                   required=True,
                   help="path to ndjson with texts to process")

    # output path
    ap.add_argument("-o", "--outpath",
                    required=True,
                    help="where results will be saved")

    # language of texts
    ap.add_argument("--lang",
                    required=False,
                    type=str, default='da',
                    help="two character ISO code of a desired language")
    # window
    ap.add_argument("--jobs",
                    required=False,
                    type=int, default=4,
                    help="number of workers to split the job between.")

    # hotfix
    ap.add_argument("--bugstring",
                required=False,
                type=bool, default=False,
                help="remove seqences of equal signs from documents?")

    # parse that
    args = vars(ap.parse_args())

    # run functions down the line
    print('[info] Importing {}'.format(args['inpath']))
    with open(args['inpath'], 'r') as f_in:
        texts = ndjson.load(f_in)

    print('[info] Clearing buggy strings.')
    if args['bugstrings']:
        texts = _delete_many_equal_signs(texts)

    print('[info] Stanza starting {} jobs'.format(args['jobs']))
    dfs_out = stanza_multicore(
        texts=texts,
        lang=args['lang'],
        n_jobs_gpu=args['jobs']
    )

    print('[info] Saving results to {}'.format(args['outpath']))
    with open(args['outpath'], "w") as f_out:
        ndjson.dump(dfs_out, f_out)


if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    main()
