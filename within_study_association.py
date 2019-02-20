#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

SCRIPTPATH = os.path.dirname(__file__)
sys.path.append(SCRIPTPATH)

import data


def parse_args():
    parser = argparse.ArgumentParser('regression.py')
    define_args(parser)
    return parser.parse_args()


def define_args(parser):
    parser.add_argument('--mirna',
                        type=str,
                        help='miRNA data')
    parser.add_argument('--mrna',
                        type=str,
                        help='mRNA data')
    parser.add_argument('--cnv',
                        type=str,
                        help='CNV data')
    parser.add_argument('--methylation',
                        type=str,
                        help='Methylation data')
    parser.add_argument('--targets',
                        type=str,
                        help='miRNA-mRNA target pairs')
    parser.add_argument('--out',
                        type=str,
                        default='regression',
                        help='Name of output file (without file suffix)')


def read_data(args):
    mirna = data.read(args.mirna)
    mrna = data.read(args.mrna)
    cnv = data.read(args.cnv)
    methylation = data.read(args.methylation)
    mrna, mirna, cnv, methylation = data.harmonize(mrna, mirna, cnv, methylation)
    print(mrna.shape, mirna.shape, cnv.shape, methylation.shape)
    return mrna, mirna, cnv, methylation


class RegressionMachine:
    def __init__(self, mrna, mirna, cnv, methylation):
        self.mrna = mrna
        self.mirna = mirna
        self.cnv = cnv
        self.methylation = methylation

    def regression_for(self, mirna, gene):
        y = self.mrna.loc[:, gene].as_matrix()
        X_mirna = self.mirna.loc[:, mirna].as_matrix()
        X_cnv = self.cnv.loc[:, gene].as_matrix()
        X_methylation = self.methylation.loc[:, gene].as_matrix()
        X = np.array([X_mirna, X_cnv, X_methylation]).T
        X = sm.add_constant(X)
        results = sm.OLS(y, X).fit()
        r = np.array([0, 1, 0, 0])
        return (results.t_test(r),
                results.t_test([0,0,1,0]),
                results.t_test([0,0,0,1]),
        )


if __name__ == '__main__':
    args = parse_args()
    mrna_data, mirna_data, cnv_data, methylation_data = read_data(args)
    rgm = RegressionMachine(mrna_data, mirna_data, cnv_data, methylation_data)
    targets = data.read_targets(args.targets)
    results = {
        'mirna_id': [],
        'gene_id': [],
        'pvalue_mirna': [],
        'beta_mirna': [],
        'pvalue_cnv': [],
        'beta_cnv': [],
        'pvalue_methyl': [],
        'beta_methyl': [],
    }
    for mirna, gene in targets:
        print('Running regression for (%s, %s) ...' % (mirna, gene))
        if mirna not in set(mirna_data.columns):
            print('No expression data for %s! Skipping.' % mirna)
            continue
        if gene not in set(mrna_data.columns):
            print('No expression data for %s! Skipping.' % gene)
            continue
        results['mirna_id'].append(mirna)
        results['gene_id'].append(gene)
        regres = rgm.regression_for(mirna, gene)
        results['beta_mirna'].append(regres[0].effect[0])
        results['pvalue_mirna'].append(regres[0].pvalue)
        results['beta_cnv'].append(regres[1].effect[0])
        results['pvalue_cnv'].append(regres[1].pvalue)
        results['beta_methyl'].append(regres[2].effect[0])
        results['pvalue_methyl'].append(regres[2].pvalue)
    results = pd.DataFrame(results)
    results['adjusted_pvalue_mirna'] = multipletests(results['pvalue_mirna'].tolist(),
                                               method='fdr_bh')[1]
    results = results[[
        'mirna_id',
        'gene_id',
        'beta_mirna',
        'pvalue_mirna',
        'adjusted_pvalue_mirna',
        'beta_cnv',
        'pvalue_cnv',
        'beta_methyl',
        'pvalue_methyl',
    ]]
    results.to_csv(args.out+'.csv.gz', compression='gzip', index=False)
