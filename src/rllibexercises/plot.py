##
##
##

import time

# default backend GtkAgg does not plot properly on Ubuntu 8.04
import matplotlib
from matplotlib import pyplot


matplotlib.use('TkAgg')


def process_events( interval=0.01 ):
    ### code taken from "matplotlib.pyplot.pause()"
    manager = pyplot._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
            ## 'show()' causes plot windows to blink/gain focus
#         show(block=False)
        canvas.start_event_loop(interval)
#     else:
#         time.sleep(interval)

    ## handle connections
    time.sleep( interval )


def draw_plot( values, fig ):
    pyplot.figure( fig.number )
    pyplot.clf()
    pyplot.plot( values )
#         plt.plot(values, 'o-')
    pyplot.gcf().canvas.draw()
    process_events()


def draw_line( values, axline ):
    xdata = range( len(values) )
    axline.set_data( xdata, values )
    ax = axline.axes
    ax.relim()
    ax.autoscale()
    fig = axline.get_figure()
    fig.canvas.draw()
    fig.canvas.flush_events()
