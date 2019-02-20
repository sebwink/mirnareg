#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

def parse_args():
    parser = argparse.ArgumentParser('cross_study_association.py')
    define_args(parser)
    return parser.parse_args()


def define_args(parser):
    parser.add_argument('--in',
                        dest='input',
                        type=str,
                        help='input pattern')
    parser.add_argument('--cancers',
                        type=str,
                        help='Study names')
    parser.add_argument('--out',
                        type=str,
                        default='association',
                        help='Name of output file (without file suffix)')


def read_input(args):
    pattern = True if len(args.input.split(',')) == 1 else False
    if pattern:
        input_files = glob.glob(args.input)
    else:
        input_files = args.input.split(',')
    names = args.cancers.split(',')
    ifiles = dict()
    for i, name in enumerate(names):
        ifile = None
        if pattern:
            for f in input_files:
                if name in f:
                    ifile = f
        else:
            ifile = input_files[i]
        if ifile is None:
            print('Invalid input file pattern %s' % args.input)
            sys.exit(-1)
        ifiles[name] = pd.read_table(ifile, compression='gzip', sep=',')
    return ifiles

def get_common_pairs(assoc):
    common = None
    for cancer in assoc:
        pairs = [
            (mirna, mrna) for _, mirna, mrna in assoc[cancer][['mirna_id', 'gene_id']].to_records()
        ]
        if common is None:
            common = set(pairs)
        else:
            common = common.intersection(set(pairs))
    return common


def chi2test(x2, df):
    return chi2.sf(x2, df)

def rec_score(pm, pp):
    if pm > pp:
        return -np.log10(pp)
    elif pm < pp:
        return np.log10(pm)
    return 0

def main(args):
    assoc = read_input(args)
    pairs = list(get_common_pairs(assoc))
    mirnas = [mirna for mirna, _ in pairs]
    mrnas = [mrna for _, mrna in pairs]
    pminus = []
    pminus_adj = []
    pplus = []
    pplus_adj = []
    for mirna, mrna in pairs:
        df = 2 * len(assoc)
        pm = chi2test(
            X2(mrna, mirna, assoc), df
        )
        pminus.append(pm)
        pp = chi2test(
            X2(mrna, mirna, assoc, reverse_ranks=True), df
        )
        pplus.append(pp)
    rec = [rec_score(pm, pp) for pm, pp in zip(pminus, pplus)]
    pminus_adj = multipletests(pminus, method='fdr_bh')[1]
    pplus_adj = multipletests(pplus, method='fdr_bh')[1]
    recadj = [rec_score(pmadj, ppadj) for pmadj, ppadj in zip(pminus_adj, pplus_adj)]
    result = pd.DataFrame({
        'mirna_id': mirnas,
        'target_id': mrnas,
        'rec': rec,
        'pminus': pminus,
        'pplus': pplus,
        'rec_adj': recadj,
        'pminus_adj': pminus_adj,
        'pplus_adj': pplus_adj
    })
    result.sort_values('rec', inplace=True)
    result = result[[
        'mirna_id',
        'target_id',
        'rec',
        'pminus', 'pplus',
        'rec_adj',
        'pminus_adj', 'pplus_adj',
    ]]
    result.to_csv(args.out+'.rec.scores.csv.gz', compression='gzip', index=False)


def rr(mrna, mirna, cancer, assoc, reverse_ranks=False):
    Lmk = L(mirna, cancer, assoc)
    rjmk = list(Lmk[Lmk['gene_id'] == mrna].index)[0] + 1
    if reverse_ranks:
        return ((len(Lmk) - rjmk + 1) - 0.5) / len(Lmk)
    else:
        return (rjmk - 0.5) / len(Lmk)

def L(mirna, cancer, assoc):
    pairs = assoc[cancer]
    Lmk = pairs[pairs['mirna_id'] == mirna].copy()
    Lmk.sort_values('beta_mirna', inplace=True)
    Lmk.index = range(len(Lmk))
    return Lmk

def X2(mrna, mirna, assoc, reverse_ranks=False):
    x2 = 0
    for cancer in assoc:
        x2 += np.log(rr(mrna, mirna, cancer, assoc, reverse_ranks))
    return -2 * x2

if __name__ == '__main__':
    args = parse_args()
    main(args)
