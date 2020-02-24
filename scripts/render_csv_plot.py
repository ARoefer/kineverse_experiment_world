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

    rec_iter = ValueRecorder('Iterations', *df.columns[6:])
    rec_iter.data = {df.columns[6]: df.T.iloc[6].to_list()}
    rec_iter.data[df.columns[7]] = (df.T.iloc[7]*1000).to_list()
    rec_iter.compute_limits()
    rec_iter.set_grid(True)
    rec_iter.set_xspace(-0.1, 20.1)
    rec_iter.set_ytitle('n iterations | ms')

    rec_err = ValueRecorder('Error Evaluation', *df.columns[2:6])
    rec_err.data = {df.columns[x]: df.T.iloc[x].to_list() for x in range(len(df.columns))[2:6]}
    rec_err.compute_limits()
    rec_err.set_grid(True)
    rec_err.set_xspace(-0.1, 20.1)

    rec_err.set_xlabels([]) # ['{:.3f} m linear $\\sigma$ \n{:.3f} rad angular $\\sigma$'.format(x, y) for x, y in df.iloc[:,:2].to_numpy()])
    #rec_iter.set_xlabels(['{:.3f} m linear $\\sigma$ \n{:.3f} rad angular $\\sigma$'.format(x, y) for x, y in df.iloc[:,:2].to_numpy()])

    draw_recorders([rec_err, rec_iter], 1.0, 8, 3).savefig('{}.png'.format(args[0][:-4]))
