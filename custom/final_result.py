import numpy as np
import os.path as osp
import os
file = np.loadtxt(osp.join(os.getcwd(),'custom_result', 'track_results.txt'))[:,:-1]
result = np.array(file)
np.savetxt(osp.join(os.getcwd(),'custom_result', 'track.txt'), result, fmt="%d %d %d %d %d %d %d %f %f")