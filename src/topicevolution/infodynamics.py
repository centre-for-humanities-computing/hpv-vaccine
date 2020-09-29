'''Novelty, Transience & Resonance of your document-topic matrix of choice.

TODO
----
calculate_ntr : IndexError
    Probably when doc ID mismatches index.
    Seems benign.

drivers : argparse
'''
import os
from itertools import chain

import numpy as np
from scipy import stats

import ndjson


def kld(p, q):
    """ KL-divergence for two probability distributions
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, (p-q) * np.log10(p / q), 0))


def jsd(p, q, base=np.e):
    '''Pairwise Jensen-Shannon Divergence for two probability distributions  
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return stats.entropy(p, m, base=base)/2. +  stats.entropy(q, m, base=base)/2.


class InfoDynamics:
    '''
    Class for estimation of information dynamics of time-dependent probabilistic document representations

    Forked from centre-for-humanities-computing/newsFluxus
    Author: Kristoffer L. Nielbo
    '''
    def __init__(self, data, time, window=3, weight=0, sort=False):
        """
        - data: list/array (of lists), bow representation of documents
        - time: list/array, time coordinate for each document (identical order as data)
        - window: int, window to compute novelty, transience, and resonance over
        - weight: int, parameter to set initial window for novelty and final window for transience
        - sort: bool, if time should be sorted in ascending order and data accordingly
        """
        self.window = window
        self.weight = weight
        if sort:
            self.data = np.array([text for _,text in sorted(zip(time, data))])
            self.time = sorted(time)
        else:
            self.data = np.array(data)
            self.time = time
        self.m = self.data.shape[0]
        
    def novelty(self, meas=kld):
        N_hat = np.zeros(self.m)
        N_sd = np.zeros(self.m)
        for i, x in enumerate(self.data):
            submat = self.data[(i - self.window):i,]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window]) + self.weight
            
            N_hat[i] = np.mean(tmp)
            N_sd[i] = np.std(tmp)

        self.nsignal = N_hat
        self.nsigma = N_sd
    
    def transience(self, meas=kld):
        T_hat = np.zeros(self.m)
        T_sd = np.zeros(self.m)
        for i, x in enumerate(self.data):
            submat = self.data[i+1:(i + self.window + 1),]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window])
            
            T_hat[i] = np.mean(tmp)
            T_hat[-self.window:] = np.zeros([self.window]) + self.weight
            T_sd[i] = np.std(tmp)
        
        self.tsignal = T_hat  
        self.tsigma = T_sd

    def resonance(self, meas=kld):
        self.novelty(meas)
        self.transience(meas)
        self.rsignal = self.nsignal - self.tsignal
        self.rsignal[:self.window] = np.zeros([self.window]) + self.weight
        self.rsignal[-self.window:] = np.zeros([self.window]) + self.weight
        self.rsigma = (self.nsigma + self.tsigma) / 2
        self.rsigma[:self.window] = np.zeros([self.window]) + self.weight
        self.rsigma[-self.window:] = np.zeros([self.window]) + self.weight

'''
Drivers
'''

def calculate(doc_top_prob, ID, window: int, out_dir=None, curb_incomplete=False):
    '''Calculate Novelty, Transience & Resonance on a single window.
    This function is wrapped in process_windows() - see it for details.
    '''

    # make sure there is a folder to save it
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    # signal calculation
    idmdl = InfoDynamics(data=doc_top_prob, time=ID,
                         window=window, weight=0, sort=False)
    idmdl.novelty(meas = jsd)
    idmdl.transience(meas = jsd)
    idmdl.resonance(meas = jsd)

    lignes = list()
    for i, doc_id in enumerate(ID):
        d = dict()
        d["doc_id"] = doc_id
        # HACK because of IndexError
        try:
            d["novelty"] = idmdl.nsignal[i] 
            d["transience"] = idmdl.tsignal[i]
            d["resonance"] = idmdl.rsignal[i]
            d["nsigma"] = idmdl.nsigma[i]
            d["tsigma"] = idmdl.tsigma[i]
            d["rsigma"] = idmdl.rsigma[i]
        except IndexError:
            print("[info] there was an Index Error, proceed with caution")
            pass

        lignes.append(d)

    if curb_incomplete:
        # keep only rows with full records
        lignes = lignes[window:-window]

    if out_dir:
        # make a filename
        filename = str(window) + 'W' + '.ndjson'
        outpath = os.path.join(out_dir, filename)

        # export
        with open(outpath, "w") as f:
            ndjson.dump(lignes, f)

    return None


def process_windows(doc_top_prob, ID, window, out_dir=None, curb_incomplete=False):
    '''Based on document-topic matrix, calcualte Novelty, Transience & Resonance.
    Entropy is calcualted relative to past and future documents.
    
    Parameters
    ----------
    doc_top_prob : list/array (of lists)
        document-topic-matrix of your topic model

    ID : list/array
        identifier, or coordinate for each document (identical order as data)
        Required by InfoDynamics

    window : int or range
        Number of documents to calculate Nov, Tra, Res against.
        See Barron, Huang, Spang & DeDeo (2018) for details.

    out_dir : str (optional)
        Directory where to save the results.

    curb_incomplete : bool (default=False)
        Delete first & last {window} rows?
        
        If false, 
        - first {window} rows will have Novelty = 0 
        - last {window} rows will have Transience = 0

        If true, these rows will not be included in the output.
        
    '''
    # if a single model is to be fitted,
    # make sure it can be "iterated"
    if isinstance(window, int):
        window = [window]

    # iterate a list of windows
    for W in chain(window):
        calculate(
            doc_top_prob=doc_top_prob,
            ID=ID,
            window=W,
            out_dir=out_dir,
            curb_incomplete=curb_incomplete
        )

    return None
