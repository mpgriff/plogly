import sys,os
import numpy as np
from apsu.io import pulseInfo
from apsu.data import ProcData
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def plot(fname):
    path,cfname = os.path.split(fname)

    all_apros = [x for x in os.listdir(path) if x.endswith('.apro')]

    i_apro = all_apros.index(cfname)

    fig,axs = plt.subplots(2,1, sharey=True, sharex=True)

    pd = ProcData.load(path+os.sep+all_apros[i_apro])
    pd.plot(axs=axs, show=False)
    axs[1].set_title(cfname)

    if len(all_apros)>1:
        if 'raw_paths' in dir(pd):
            axs[1].text(0.99, 0.95, 'click decay to review dat file', ha='right', va='top', transform=axs[1].transAxes)

        class Index:
            def __init__(self, i_apro):
                self.i_apro = i_apro
                self.pd = ProcData.load(path+os.sep+all_apros[self.i_apro])

            @property
            def fname(self):
                cpath = path+os.sep+all_apros[self.i_apro]
                cpath,cfname = os.path.split(cpath)
                return cfname

            def next(self, event):
                if self.i_apro+1 < len(all_apros):
                    self.i_apro += 1
                else:
                    print('no more apros')
                axs[0].clear()
                axs[1].clear()
                self.pd = ProcData.load(path+os.sep+all_apros[self.i_apro])
                self.pd.plot(axs=axs, show=False)
                axs[1].set_title(self.fname)
                plt.draw()

            def prev(self, event):
                if self.i_apro-1 > 0:
                    self.i_apro -= 1
                else:
                    print('no more apros')
                axs[0].clear()
                axs[1].clear()
                self.pd = ProcData.load(path+os.sep+all_apros[self.i_apro])
                self.pd.plot(axs=axs, show=False)
                axs[1].set_title(self.fname)
                plt.draw()

            def data_pick(self, event):
                i = -1
                if event.inaxes in [axs[1]]: real_click=True

                if real_click:
                    j = int(np.round(event.xdata, 0))
                    rx_data = self.pd.raw_paths[j]
                    from plot_dat import plot
                    plot(rx_data, line=f'C{j}-')

        callback = Index(i_apro)
        axprev = plt.axes([0.8, 0.025, 0.09, 0.075])
        axnext = plt.axes([0.9, 0.025, 0.09, 0.075])
        bnext = Button(axnext, 'next')
        bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'prev')
        bprev.on_clicked(callback.prev)
        fig.canvas.callbacks.connect('button_press_event', callback.data_pick)

    plt.show()

if __name__ == "__main__":
    fname = sys.argv[1]
    plot(fname)
