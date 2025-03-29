import pandas as pd
import numpy as np

from macrel.macrel_features import normalize_seq, compute_all


def model_predict(model1, data):
    features = data.iloc[:, 3:]

    if len(features):
        amp_prob = model1.predict_proba(features).T[0]
    else:
        amp_prob = np.array([])

    return amp_prob


def fasta_features(input_list):
    '''Compute features for all sequences in a given FASTA file'''
    seqs = []
    headers = []
    features = []
    for index, seq in enumerate(input_list):
        seq = normalize_seq(seq)
        seqs.append(seq)
        headers.append('AP' + str(index))
        features.append(compute_all(seq))

    features = pd.DataFrame(features, index=headers, columns=[
        "tinyAA",
        "smallAA",
        "aliphaticAA",
        "aromaticAA",
        "nonpolarAA",
        "polarAA",
        "chargedAA",
        "basicAA",
        "acidicAA",
        "charge",
        "pI",
        "aindex",
        "instaindex",
        "boman",
        "hydrophobicity",
        "hmoment",
        "SA.Group1.residue0",
        "SA.Group2.residue0",
        "SA.Group3.residue0",
        "HB.Group1.residue0",
        "HB.Group2.residue0",
        "HB.Group3.residue0",
    ])
    features.insert(0, 'group', 'Unk')
    features.insert(0, 'sequence', seqs)
    return features


# def do_get_examples(args):
#     try:
#         from urllib.request import urlretrieve
#     except:
#         from urllib2 import urlretrieve
#
#     DATA_FILES = [
#         'excontigs.fna.gz',
#         'expep.faa.gz',
#         'R1.fq.gz',
#         'R2.fq.gz',
#         'ref.faa.gz',
#     ]
#     BASEURL = 'https://github.com/BigDataBiology/macrel/raw/master/example_seqs/'
#     if path.exists('example_seqs') and not args.force:
#         error_exit(args, 'example_seqs/ directory already exists')
#     makedirs('example_seqs', exist_ok=True)
#     for f in DATA_FILES:
#         print('Retrieving {}...'.format(f))
#         urlretrieve(BASEURL + f, 'example_seqs/' + f)


def macrel_predictor(fasta_file,model1):
    fs = fasta_features(fasta_file)
    prediction = model_predict(model1, fs)
    return prediction


