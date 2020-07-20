from topicevolution.infodynamics import InfoDynamics
from topicevolution.entropies import jsd


def get_doc_top_gensim(lda_model_path, bows):
    '''
    Extract a document-topic matrix from your Gensim LDA model.

    Parameters
    ----------
    lda_model_path : str
        path to a .model file

    bows : to be deleted & worked around
    '''
    # load model
    model = models.LdaModel.load(lda_model_path)
    # load bows
    # TODO

    # GENSIM doc-top matrix
    # min topic prob = 0 for uniform array shape
    doc_top = [model.get_document_topics(doc, minimum_probability=0)
               for doc in model[bows]]

    # unnest (n topic, prob) tuples
    doc_top_prob = [
        [prob for i, prob in doc]
        for doc in doc_top
    ]

    # TODO: export document-topic matrix
#     doctopics_df = pd.DataFrame([])
#     for doc in doc_top_prob:
#         doc = pd.DataFrame(doc).transpose() 
#         doctopics_df = doctopics_df.append(doc.iloc[1,:],
#                                            ignore_index=True)
#     doc_top_matrix_path = os.path.join(
#         out_dir, 'data',
#         lda_model_path.replace('.model',''), "T_doctopmat.csv"
#     )
#     doctopics_df.to_csv('topic_models/batch_200621/topic_evolution/22T_doctop.csv')

    return doc_top_prob


def get_doc_top_sklearn(lda_model_path):
    '''
    Extract a document-topic matrix from your Sklearn LDA, or guidedlda model.

    Parameters
    ----------
    lda_model_path : str
        path to a .model file
    '''

    return None # doc_top_prob


def summarie_doc_top():
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
