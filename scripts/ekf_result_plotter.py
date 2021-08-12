import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

import argparse

from kineverse.visualization.plotting import ValueRecorder, draw_recorders, split_recorders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotter for EKF data')

    parser.add_argument('--results',  '-r', type=str, help='Results file')
    parser.add_argument('--stats',  '-s', type=str, help='Stats file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output file')

    args = parser.parse_args()
    if args.output is None:
        args.output = '{}.png'.format(args.results.split('.')[0])

    df_r = pd.read_csv(args.results)
    df_s = pd.read_csv(args.stats)

    n_state = (len(df_r.columns) - 2) / 3

    noise_labels = []
    estimate_data_points = []

    for l_sd, a_sd in zip(df_r.lin_std.unique(), df_r.ang_std.unique())[1:]:
        noise_labels.append('{:3.3f} m linear $\\sigma$\n{:3.3f} rad angular $\\sigma$'.format(l_sd, a_sd))

        sf = df_r[df_r.lin_std == l_sd]
        gt_states    = sf.iloc[:, 2:n_state + 2].to_numpy()
        final_states = sf.iloc[:, n_state + 2:2 * n_state + 2].to_numpy()
        final_vars   = sf.iloc[:, 2 * n_state + 2:3 * n_state + 2].to_numpy()

        delta = np.abs(gt_states - final_states)
        estimate_data_points.append([delta.mean(), delta.std(), delta.min(), delta.max()])

    columns  = ['Mean Error', 'SD Error', 'Min Error', 'Max Error']
    rec_err = ValueRecorder('Error Evaluation', *columns)
    rec_err.data = {x: v for x, v in zip(columns, np.vstack(estimate_data_points).T)}
    rec_err.compute_limits()
    rec_err.set_grid(True)
    rec_err.set_xspace(-0.1, len(estimate_data_points) - 0.9)
    rec_err.set_xlabels(noise_labels)
    rec_err.set_marker('.')


    np_p = df_s.to_numpy() * 1000
    iter_columns = ['Mean Duration', 'SD Duration'] # , 'Min Duration', 'Max Duration']
    rec_iter = ValueRecorder('Iterations', *iter_columns)
    rec_iter.data = {iter_columns[0]: np_p.mean(axis=0),
                     iter_columns[1]: np_p.std(axis=0),}
                     # iter_columns[2]: np_p.min(axis=0),
                     # iter_columns[3]: np_p.max(axis=0)}
    rec_iter.compute_limits()
    rec_iter.set_grid(True)
    # rec_iter.set_xspace(-0.1, df_performance['DoF'].max() + 0.1)
    rec_iter.set_ytitle('ms')
    rec_iter.set_xtitle('Degrees of Freedom')
    rec_iter.set_xlabels([str(int(x)) for x in range(1, np_p.shape[1] + 1)])

    draw_recorders([rec_err, rec_iter], 1.0, 8, 3).savefig(args.output)
