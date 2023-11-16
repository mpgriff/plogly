from plog import Borehole
import matplotlib.pyplot as plt
import numpy as np
import sys, os

outfolder = 'C:/Users/au646858/_GIT/plog/examples/endelave/'
if not os.path.exists(outfolder): os.mkdir(outfolder)


import shapefile
shape = shapefile.Reader(r"Z:\2023_Dart\Endelave\Shape\Endelave_Boreholes.shp")
dart_folder = 'Z:/2023_Dart/Endelave/Processed/Export/'

with open(outfolder+'endelave_boreholes.csv', 'wt') as f:
    for pnt in shape.shapeRecords():
        x,y,bhnum = pnt.record[1], pnt.record[2], pnt.record[4]
        bh = Borehole.dart(dart_folder+bhnum+'/'+bhnum)
        outname = outfolder+bhnum+'.plog'
        # bh.plot()
        # plt.show()
        bh.save(outname)
        row = [str(x), str(y), bhnum, f'\"{outname}\"']
        f.write(','.join(row)+'\n')
