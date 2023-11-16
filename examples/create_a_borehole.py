from plog import Log, Borehole
import matplotlib.pyplot as plt
import numpy as np

lg1 = Log.geology(['sand', 'clay', 'silt', 'sand'], [0, 5, 15, 25], [5, 15, 25, 40], color_dictionary='terrain')
lg2 = Log.standard(np.random.randn(31), np.linspace(0., 40, 31)[1:], 'rho', units='Ωm')

bh = Borehole([lg1, lg2])
bh.plot()
plt.show()
bh.save('test_borehole.plog')
