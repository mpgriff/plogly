""" 
    2023/05/24 - Matt Griffiths

    This script finds the relevant plotting routine associated with the a certain file extension.
    The main reason for doing things this way, is a batch script is required to activate the python environment to default open from programs like qgis.
    This way only a single batch script (apsu_opener.cmd) is necessary along with this script for finding the appropriate plotting routine.
    
"""
try:
    import sys, os
    import matplotlib.pyplot as plt

    fname = sys.argv[1]

    from dill import load
    with open(fname, 'rb') as f:
        obj = load(f)
    ax = obj.plot()
    fig = ax.flatten()[0].get_figure()
    fig.suptitle(fname)
    plt.tight_layout()
    plt.show()


except Exception as e:
    import traceback
    import sys,os
    error_log = fname+'.plotting.err'
    print(str(e))
    print(traceback.format_exc())
    with open(error_log, 'w') as f:
        f.write(str(e))
        f.write(traceback.format_exc())
