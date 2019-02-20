import pandas as pd


def read(path):
    data = pd.read_table(path, compression='gzip', sep=',')
    return data.set_index('id')


def harmonize(*args):
    common_samples = set(args[0].columns).intersection(*[
        set(df.columns) for df in args[1:]
    ])
    common_samples = list(common_samples)
    return [df[common_samples].T for df in args]


def read_targets(path):
    targets = pd.read_table(path, compression='gzip', sep=',')
    targets['mirna_id'] = [ID.lower() for ID in targets['mirna_id']]
    return targets[['mirna_id', 'gene_id']].to_records(index=False).tolist()
