'''Novelty, Transience & Resonance of your document-topic matrix of choice.

TODO
----
- calculate_ntr: IndexError
'''

from gensim.models import LdaModel
from gensim.corpora import Dictionary

from topicevolution.infodynamics import InfoDynamics
from topicevolution.entropies import jsd


def summarzie_doc_top():
    '''
    Summarizes probability distributions from a document-topic matrix
    into a distribution, corresponding to chunks of certain temporal length.

    E.g. get a probability distribution of 10 minutes of facebook posts.

    NOTE: maybe not even needed in HPV, but deffinitely yes in HOPE!

    Parameters
    ----------
    ???
    '''
    return None


def calculate_ntr(doc_top_prob, time, window, out_dir=None):
    '''Based on document-topic matrix, calcualte Novelty, Transience & Resonance.
    
    Parameters
    ----------
    doc_top_prob : list/array (of lists)
        document-topic-matrix of your topic model

    time : list/array
        Time coordinate for each document (identical order as data)
        Required by InfoDynamics

    window : int
        Number of documents to calculate Nov, Tra, Res against.
        See Barron, Huang, Spang & DeDeo (2018) for details.

    out_path : str (optional)
        Where to save the results.
    '''
    # signal calculation
    idmdl = InfoDynamics(data=doc_top_prob, time=time,
                         window=window, weight=0, sort=False)
    idmdl.novelty(meas = jsd)
    idmdl.transience(meas = jsd)
    idmdl.resonance(meas = jsd)

    lignes = list()
    for i, time in enumerate(dates):
        d = dict()
        d["date"] = time
        # HACK because of IndexError
        try:
            d["novelty"] = idmdl.nsignal[i] 
            d["transience"] = idmdl.tsignal[i]
            d["resonance"] = idmdl.rsignal[i]
            d["nsigma"] = idmdl.nsigma[i]
            d["tsigma"] = idmdl.tsigma[i]
            d["rsigma"] = idmdl.rsigma[i]
        except IndexError:
            print("[info] there was an Index Error, but we're taking care of the situation")
            pass
        lignes.append(d)

    if outpath:
        with open(outpath, "w") as f:
            ndjson.dump(lignes, f)

    return lignes
