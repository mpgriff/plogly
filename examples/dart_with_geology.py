from plog import Borehole, Dart, Log
import matplotlib.pyplot as plt
import numpy as np
import sys, os

outfolder = 'C:/Users/au646858/'
if not os.path.exists(outfolder): os.mkdir(outfolder)

B6geo = Log.geology(['Torvemuld', 'sand ler', 'gra sand', 'brown sand', 'gra fine sand', 'sand silt ler', 'gra moronler', 'til', 'sand', 'gra sand', 'sand'],
                         [0, 0.3, 0.6, 1.6, 2.3, 4.4, 7.1, 11.1, 15.2, 16, 16.6], [0.3, 0.6, 1.6, 2.3, 4.4, 7.1, 11.1, 15.2, 16, 16.6, 19.6], color_dictionary='terrain', name='B6')
bh  = Dart('O:/Nat-Tech_HGG-Projects/2023_Dart/Endelave/Processed/Export/B6/B6')
bh.logs.insert(0, B6geo)

#fig,axs = plt.subplots(1,6, width_ratios=[0.5, 1, 2, 2, 1, 1], sharey=True)
fig = bh.plot()
fig.show()
# plt.tight_layout()
# plt.show()