import matplotlib.pyplot as plt
import numpy as np

def plot_word_dist(info, words, start_year, end_year, one_minus=False, legend_loc='upper left'):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plot_info = {}
    for word in words:
        plot_info[word] = info[word]
    for title, data_dict in plot_info.iteritems():
        x = []; y = []
        for year, val in data_dict.iteritems():
            if year >= start_year and year <= end_year:
                x.append(year)
                if one_minus:
                    val = 1 - val
                y.append(val)
        color = colors.pop()
        plt.plot(x, smooth(np.array(y)), color=color)
        plt.scatter(x, y, marker='.', color=color)
    plt.legend(plot_info.keys(), loc=legend_loc)
    return plt

def plot_word_basic(info, words, start_year, end_year, datatype):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plot_info = {}
    for word in words:
        plot_info[word] = info[word]
    for title, data_dict in plot_info.iteritems():
        x = []; y = []
        for year, val in data_dict[datatype].iteritems():
            if year >= start_year and year <= end_year:
                x.append(year)
                y.append(val)
        color = colors.pop()
        plt.plot(x, smooth(np.array(y)), color=color)
        plt.scatter(x, y, marker='.', color=color)
    plt.legend(plot_info.keys())
    plt.show()
 
def plot_basic(plot_info, start_year, end_year):
    for title, data_dict in plot_info.iteritems():
        x = []; y = []
        for year, val in data_dict.iteritems():
            if year >= start_year and year <= end_year:
                x.append(year)
                y.append(val)
        plt.plot(x, y)
    plt.legend(plot_info.keys())
    plt.show()

import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    y = y[(window_len/2 - 1):-(window_len/2 + 1)]
    return y
