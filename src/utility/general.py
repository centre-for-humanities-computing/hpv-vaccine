'''
General utility scripts
'''
import os

import ndjson
import pandas as pd


def validate_input(texts) -> list:
    '''
    Make sure texts are in the right format
    before training PMI-SVD embeddings.

    If no exceptions are raised, returns a list in
    the following format: output[document][word]
    '''
    # check if empty
    if not texts:
        raise ValueError('"texts" input is empty!')

    # check for string input
    elif isinstance(texts, str):
        # if there is a single line, parse as one doc
        if texts.count('/n') == 0:
            output = [[word.lower() for word in texts.split()]]
        # if multiple lines, each line is a new doc
        else:
            texts = texts.split('\n')
            output = [[word for word in doc.split()]
                      for doc in texts]

    # check for input list
    elif isinstance(texts, list):
        # if the list is nested
        if all(isinstance(doc, list) for doc in texts):
            # validate that all items of sublists are str
            unnest = [word for doc in texts for word in doc]
            if all(isinstance(item, str) for item in unnest):
                output = texts
            else:
                raise ValueError(
                    "input: all items in texts[i][j] must be strings")
        # if the list is not nested
        elif all(isinstance(doc, str) for doc in texts):
            output = [[word for word in doc.split()]
                      for doc in texts]
        # if any texts[i] are other types throw error
        else:
            raise ValueError("input: incompatible data type in texts[i]")

    # break when input is neither str or list
    else:
        raise ValueError('texts must be str or list')

    return output


def load_data(ndjson_path):
    '''
    Read a preprocessed file & convert to ttx format.
    '''
    with open(ndjson_path, 'r') as f:
        obj = ndjson.load(f)

    obj_dfs = [pd.DataFrame(dat) for dat in obj]

    return obj_dfs


def make_folders(out_dir):
    '''
    Create folders for saving many models
    
    out_dir : str
        path to export models to
    '''

    # create main folder
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # get output paths
    report_dir = os.path.join(out_dir, "report_lines", "")
    model_dir = os.path.join(out_dir, "models", "")
    plot_dir = os.path.join(out_dir, "plots", "")
    doctop_dir = os.path.join(out_dir, "doctop_mats", "")

    # create sub-folders
    for folder in [report_dir, model_dir, plot_dir, doctop_dir]:
        # check if dir already exists
        if not os.path.exists(folder):
            os.mkdir(folder)

    return None


def export_serialized(df, column='text', path=None):
    '''
    Serialize column to a dictionary,
    where keys are ID and values are col.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to unpack

    column : str (default: 'text')
        name of df's column to export

    path : str, optional
        where to save the resulting .ndjson object
    '''
    
    # get ID column
    df_id = (
        df
        .reset_index()
        .rename(columns={'index': 'ID'})
    )

    # convert data to list of dicts
    serial_output = []
    for i, row in df_id.iterrows():
        doc = {'ID': row['ID'], column: row[column]}
        serial_output.append(doc)

    # if path is specified, save & be silent
    if path:
        with open(path, 'w') as f:
            ndjson.dump(serial_output, f)
        return None

    # if no path, return list of dicts
    else:
        return serial_output


def compile_report(report_dir):
    '''
    Join partial reports from LDA training into one DF.
    Returns a DF sorted in descending order by avg topic coherence in that model.

    Parameters
    ----------
    report_dir : str
        path to directory, where reports are saved.
        Report are serialized tuples in .ndjson format.
        See lda training scripts for details.
    '''
    # get a list of paths to import
    report_paths = []
    for file in os.listdir(report_dir):
        if file.endswith(".ndjson"):
            # tuple with whole path and file name
            path_and_file = tuple([report_dir + file, file])
            # append both
            report_paths.append(path_and_file)

    # iterate through paths, converting them into DF rows
    dfs = pd.DataFrame([])
    for path, model_name in report_paths:
        # load the tuple
        with open(path, 'rb') as f:
            report = ndjson.load(f)

        # convert to a df row
        report_row = pd.DataFrame([report],
                                  columns=['n_top', 'alpha', 'eta',
                                           'training_time', 'coh_score',
                                           'coh_topic'])
        # add model name info
        report_row.insert(0, 'model', model_name.replace('.ndjson', ''))
        # compile report
        dfs = dfs.append(report_row)

    # sort models
    dfs = (dfs
           .sort_values(by='coh_score', ascending=False)
           .reset_index()
           .drop('index', 1))

    return dfs
