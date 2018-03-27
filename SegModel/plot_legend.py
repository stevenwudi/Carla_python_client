# plot the legend of Cityscape
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt

from SegModel.city_scape_info import CITYSCAPE_PALETTE, labels

patches = []
for label in labels:
    if label.trainId != 255:
        patches.append(mpatches.Patch(color='#%02x%02x%02x' % tuple(CITYSCAPE_PALETTE[label.trainId])
                                      , label=label.name))
plt.legend(handles=patches)
plt.savefig('./SegModel/legend.png', dpi=140, bbox_inches='tight')
