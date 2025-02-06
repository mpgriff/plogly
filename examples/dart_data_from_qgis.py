from plog import Borehole, Dart, Log
import matplotlib.pyplot as plt
import numpy as np
import sys, os

outfolder = 'C:/Users/au646858/'
if not os.path.exists(outfolder): os.mkdir(outfolder)

bhgeo={ 'B6':Log.geology(['Torvemuld', 'sand ler', 'gra sand', 'brown sand', 'gra fine sand', 'sand silt ler', 'gra moronler', 'til', 'sand', 'gra sand', 'sand'],
                         [0, 0.3, 0.6, 1.6, 2.3, 4.4, 7.1, 11.1, 15.2, 16, 16.6], [0.3, 0.6, 1.6, 2.3, 4.4, 7.1, 11.1, 15.2, 16, 16.6, 19.6], color_dictionary='terrain', name='B6'),
        'B7':Log.geology(['sand', 'brown sand', 'gra sand', 'gytje', 'ler', 'sand w/ ler', 'sand', 'sand w/ ler', 'til'],
                         [0., 0.2, 1.1, 1.8, 3.2, 3.6, 9, 12.4, 13.], [0.2, 1.1, 1.8, 3.2, 3.6, 9, 12.4, 13., 20], color_dictionary='terrain', name='B7'),
        'B5':Log.geology(['muld', 'sand', 'ler', 'brown sand', 'sand', 'sand silt', 'moronler'],
                         [0., 0.2, 0.6, 1.6,  3.2, 6.3, 10.2],
                         [0.2, 0.6, 1.6,  3.2, 6.3, 10.2, 15.], color_dictionary='terrain', name='B5')}

bh = Borehole([bhgeo['B6'], bhgeo['B5']])
bh.save(outfolder+'geolog.plog')

import shapefile
shape = shapefile.Reader(r"Z:\2023_Dart\Endelave\Shape\Endelave_Boreholes.shp")
dart_folder = 'Z:/2023_Dart/Endelave/Processed/Export/'

with open(outfolder+'endelave_boreholes.csv', 'wt') as f:
    for pnt in shape.shapeRecords():
        x,y,bhnum = pnt.record[1], pnt.record[2], pnt.record[4]
        
        bh = Dart(dart_folder+bhnum+'/'+bhnum)
        print(bhnum)
        if bhnum in bhgeo.keys():
            bh.logs.insert(0, bhgeo[bhnum])

        outname = outfolder+bhnum+'.plog'
        bh.plot()
        bh.save(outname)
        row = [str(x), str(y), bhnum, f'\"{outname}\"']
        f.write(','.join(row)+'\n')

plt.show()