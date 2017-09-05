import caffe
import surgery, score
import sys
import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))
from pylab import *

outPut = open('/snapshot/train/vAc_vIU_vLs_tLs.csv', 'w')


weights = '/data/model/VGG_ILSVRC_16_layers.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)

base_net = caffe.Net('/data/model/vgg16.prototxt', '/data/model/VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)
surgery.transplant(solver.net, base_net)
del base_net

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('/imgNamVAL.txt', dtype=str)

#load snapshot
#solver.restore('/snapshot/train_iter_3600.solverstate')



# init vars to train and  store results
size_intervals = 20 #4000 No of iterations between each validation and plot
num_intervals = 8000  #2500 No of times to validate and plot
total_iterations = size_intervals * num_intervals # 2500*4000 = 10.000.000 total iterations

# set plots data
train_loss = np.zeros(num_intervals)
val_loss = np.zeros(num_intervals)
val_acc = np.zeros(num_intervals)
val_iu = np.zeros(num_intervals)
it_axes = (arange(num_intervals) * size_intervals) + size_intervals

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss (b) - val loss (r)')
ax2.set_ylabel('val accuracy (y) - val iu (g)')
ax2.set_autoscaley_on(False)
ax2.set_ylim([0, 1])

for it in range(num_intervals):

    solver.step(size_intervals)
    # solver.net.forward()

    # Test with validation set every 'size_intervals' iterations
    [loss, acc, iu] = score.seg_tests(solver, False, val, layer='score')
    val_acc[it] = acc
    val_iu[it] = iu
    val_loss[it] = loss
    train_loss[it] = solver.net.blobs['loss'].data

    outPut.write('%f;%f;%f;%f\n' % (val_acc[it],val_iu[it],val_loss[it],train_loss[it]))

    # Plot results
    if it > 0:
        ax1.lines.pop(1)
        ax1.lines.pop(0)
        ax2.lines.pop(1)
        ax2.lines.pop(0)

    ax1.plot(it_axes[0:it+1], train_loss[0:it+1], 'b') #Training loss averaged last 20 iterations
    ax1.plot(it_axes[0:it+1], val_loss[0:it+1], 'r')    #Average validation loss
    ax2.plot(it_axes[0:it+1], val_acc[0:it+1], 'y') #Average validation accuracy (mean accuracy of text and background)
    ax2.plot(it_axes[0:it+1], val_iu[0:it+1], 'g')  #Average intersecction over union of score-groundtruth masks
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt_dir = '/snapshot/train/training-' + str(solver.iter) + '.png' #Save graph every "size intervals"
    savefig(plt_dir, bbox_inches='tight')

outPut.close()
