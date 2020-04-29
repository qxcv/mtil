"""Collate .csv files from mixed runs."""

import os
import sys

import click
from milbench.benchmarks import EnvName
import numpy as np
import pandas as pd


def cell_formatter(row):
    mu, c95a, c95b = row['mean_score'], row['ci95_lower'], row['ci95_upper']
    assert np.allclose(mu - c95a, c95b - mu), (mu - c95a, c95b - mu)
    cell = r'%.2f ($\pm$%.2f)' % (mu, mu - c95a)
    return cell


def simplify_column_name(col_name):
    name = EnvName(col_name)
    return name.name_prefix, name.demo_test_spec.strip('-')


@click.option("--out-path",
              default=None,
              help="file path to write to (if not stdout)")
@click.command(
    help="Collect .csv files produced by `mtbc testall` and present "
    "them in a series of legible tables")
@click.argument("csvs", nargs=-1, required=True)
def cli(csvs, out_path):
    if out_path is not None:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_fp = open(out_path, 'w')
    else:
        out_fp = sys.stdout

    # join all data files together
    data = pd.concat(pd.read_csv(file_path) for file_path in csvs)
    data = data.set_index(np.arange(len(data.index)))
    cell_contents = data.apply(cell_formatter, axis=1)
    data['cell_contents'] = cell_contents
    data['Algo'] = data['latex_alg_name']
    for demo_env_name, group_inds in data.groupby('demo_env').groups.items():
        subset = data.iloc[group_inds]
        pivot = subset.pivot(index='Algo',
                             columns='test_env',
                             values='cell_contents')
        new_columns = []
        common_name = None
        for column in pivot.columns:
            prefix, new_name = simplify_column_name(column)
            if common_name is None:
                common_name = prefix
            else:
                assert common_name == prefix, (prefix, common_name)
            new_columns.append(new_name)
        pivot.columns = new_columns
        latex_str = pivot.to_latex(escape=False,
                                   column_format='l' + 'c' * len(new_columns))
        print(r'\subsection{' + common_name + '}\n', file=out_fp)
        print(latex_str, file=out_fp)


if __name__ == '__main__':
    try:
        with cli.make_context(sys.argv[0], sys.argv[1:]) as ctx:
            result = cli.invoke(ctx)
    except click.ClickException as e:
        e.show()
        sys.exit(e.exit_code)
    except click.exceptions.Exit as e:
        if e.exit_code == 0:
            sys.exit(e.exit_code)
        raise
