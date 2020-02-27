#!/usr/bin/env python
import sys
import pandas as pd
import numpy  as np

from kineverse.visualization.plotting import ValueRecorder, draw_recorders, split_recorders

if __name__ == '__main__':
    
    args = sys.argv[1:]

    if len(args) < 1:
        print('csv file needed')
        exit(0)

    df = pd.read_csv(args[0])

    df_error = df[df.Poses == 60]

    df_performance = pd.DataFrame([df[df.DoF == x].mean() for x in set(df['DoF'])])

    iter_columns = ['Mean Iterations','Iteration Duration']
    rec_iter = ValueRecorder('Iterations', *iter_columns)
    rec_iter.data = {iter_columns[0]: df_performance[iter_columns[0]].to_list()}
    rec_iter.data[iter_columns[1]] = (df_performance[iter_columns[1]]*1000).to_list()
    rec_iter.compute_limits()
    rec_iter.set_grid(True)
    # rec_iter.set_xspace(-0.1, df_performance['DoF'].max() + 0.1)
    rec_iter.set_ytitle('n iterations | ms')
    rec_iter.set_xtitle('Degrees of Freedom')
    rec_iter.set_xlabels([str(int(x)) for x in df_performance['DoF']])

    columns  = ['Mean Error', 'SD Error', 'Min Error', 'Max Error']
    rec_err = ValueRecorder('Error Evaluation', *columns)
    rec_err.data = {x: df_error[x].to_list() for x in columns}
    rec_err.compute_limits()
    rec_err.set_grid(True)
    rec_err.set_xspace(-0.1, len(df_error) - 0.9)
    rec_err.set_xlabels(['{:.3f} m linear $\\sigma$ \n{:.3f} rad angular $\\sigma$'.format(x, y) for x, y in zip(df_error['Linear SD'], df_error['Angular SD'])])
    rec_err.set_marker('.')

    # rec_err.set_xlabels([]) # ['{:.3f} m linear $\\sigma$ \n{:.3f} rad angular $\\sigma$'.format(x, y) for x, y in df.iloc[:,:2].to_numpy()])
    #rec_iter.set_xlabels(['{:.3f} m linear $\\sigma$ \n{:.3f} rad angular $\\sigma$'.format(x, y) for x, y in df.iloc[:,:2].to_numpy()])

    draw_recorders([rec_err, rec_iter], 1.0, 8, 3).savefig('{}.png'.format(args[0][:-4]))
