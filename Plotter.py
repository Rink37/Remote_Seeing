import numpy as np
import Data_processor as dp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

#plot and construct_subplot are default functions which take correctly formatted arguments and produce plots

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams.update({"text.usetex":True})
matplotlib.rcParams.update({'font.size': 12})

def plot(data, layout):
    if layout == []:
        layout = np.array([[1,1]]*len(data))
    length = 0
    for i in layout:
        length = length+i[0]*i[1]
    sl = np.sqrt(length)
    shape = [0,0]
    shape[0] = int(sl)
    shape[1] = int(length-sl)
    if shape[1] == 0:
        shape[1] = 1
    pos = np.zeros([len(layout), 4])
    occ = np.zeros(shape)
    xd = 0
    yd = 0
    cc = True
    for i in range(len(layout)):
        for x in range(shape[1]):
            for y in range(shape[0]):
                if occ[y,x] == 0 and cc:
                    xd = x
                    yd = y
                    cc = False
        cc = True
        pos[i,0] = yd
        pos[i,1] = yd+(layout[i])[0]
        pos[i,2] = xd
        pos[i,3] = xd+(layout[i])[1]
        occ[int(pos[i,0]):int(pos[i,1]), int(pos[i,2]):int(pos[i,3])] = 1
    fig = plt.figure(figsize = (6, 3), dpi = 300, constrained_layout = True)
    gs = fig.add_gridspec(shape[0], shape[1])
    for i in range(len(layout)):
        p = pos[i]
        locals()['ax'+str(i)] = fig.add_subplot(gs[int(p[0]):int(p[1]), int(p[2]):int(p[3])])
        dat = data[i]
        construct_subplot(dat, locals()['ax'+str(i)])
    plt.show()
     
def construct_subplot(data, ax):
    if 'xs' not in data and 'ys' not in data and 'image' not in data:
        print('No plottable data found')
        return
    elif 'xs' not in data and 'ys' not in data and 'image' in data:
        im = ax.imshow(data['image'])
        if 'ylabel' in data:
            ax.set_ylabel(data['ylabel'])
        if 'xlabel' in data:
            ax.set_xlabel(data['xlabel'])
        if 'title' in data:
            plt.title(data['title'])
        if 'xax' in data:
            ax.xaxis.set_visible(data['xax'])
        if 'yax' in data:
            ax.yaxis.set_visible(data['yax'])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size = "5%", pad = 0.05)
        plt.colorbar(im, cax = cax)
    else:
        plot_hist = False
        if 'image' in data:
            im = ax.imshow(data['image'])
        xs = enforce_shape(data['xs'])
        ys = enforce_shape(data['ys'])
        if xs.shape != ys.shape:
            print('Data is not in arrays with the same shape')
            return
        if 'xerr' in data:
            xerror = enforce_shape(data['xerr'])
        else:
            xerror = [None]*xs.shape[0]
        if 'yerr' in data:
            yerror = enforce_shape(data['yerr'])
        else:
            yerror = [None]*ys.shape[0]
        if 'method' in data:
            if isinstance(data['method'], str):
                if data['method'] == 'scatter':
                    l = 0
                    ls = [l]*xs.shape[0]
                    m = ['x']*xs.shape[0]
                elif data['method'] == 'line':
                    l = 1
                    ls = [l]*xs.shape[0]
                    m = [None]*xs.shape[0]
                elif data['method'] == 'hist':
                    plot_hist = True
                    l = 0
                    ls = [l]*xs.shape[0]
                    m = [None]*xs.shape[0]
                else:
                    l = 1
                    ls = [l]*xs.shape[0]
                    m = [None]*xs.shape[0]
            else:
                ls = [1]*xs.shape[0]
                m = [1]*xs.shape[0]
                for i in range(len(data['method'])):
                    if data['method'][i] == 'scatter':
                        ls[i] = 0
                        m[i] = 'x'
                    elif data['method'][i] == 'line':
                        ls[i] = 1
                        m[i] = None
                    elif data['method'][i] == 'hist':
                        plot_hist = True
                    else:
                        ls[i] = 1
                        m[i] = None
        else:
            l = 1
            ls = [l]*xs.shape[0]
            m = [None]*xs.shape[0]
        if 'linestyle' in data:
            if isinstance(data['linestyle'], str):
                linestyle = [data['linestyle']]*xs.shape[0]
            else:
                linestyle = []
                for i in range(len(data['linestyle'])):
                    linestyle += [data['linestyle'][i]]
        else:
            linestyle = ['-']*xs.shape[0]
        if 'colour' in data:
            col = data['colour']
        else:
            col = [None]*xs.shape[0]
        if 'temps' in data:
            temps = data['temps']
            if len(temps) > 1:
                norm = plt.Normalize(np.array(temps[len(temps)-1]).min(), np.array(temps[0]).max()+15)
            else:
                norm = plt.Normalize(np.array(temps).min(), np.array(temps).max()+15)
            for i in range(xs.shape[0]):
                points = np.array([np.array(xs[i]), np.array(ys[i])]).T.reshape(-1,1,2)
                segments = np.concatenate([points[:-1], points[1:]], axis = 1)
                lc = LineCollection(segments, cmap = 'plasma', norm=norm)
                lc.set_array(np.array(temps)[i])
                lc.set_linewidth(l)
                if i != 0 and i != xs.shape[0]-1:
                    lc.set_alpha(0)
                if not plot_hist:
                    if temps[i][0] == temps[i][-1]:
                        col[i] = matplotlib.cm.get_cmap('plasma')(norm(temps[i][0]))
                    markers, caps, bars = ax.errorbar(xs[i], ys[i], yerr = yerror[i], xerr = xerror[i], elinewidth = 1, lw = ls[i], marker = m[i], capsize = 3, color = col[i], linestyle = linestyle[i])
                    if temps[0][0] != temps[0][-1]:
                        markers.set_alpha(0)
                        [bar.set_alpha(1) for bar in bars]
                        [cap.set_alpha(1) for cap in caps]
                        line = ax.add_collection(lc)
                else:
                    col[i] = matplotlib.cm.get_cmap('plasma')(norm(temps[i][0]))
                    ax.hist(ys[i], orientation = 'horizontal', color = col[i])
            if 'hidecb' not in data:
                if plot_hist:
                    line = ax.add_collection(lc)
                cbar = plt.colorbar(line, ax=ax)
                cbar.set_label(r'$T_{p1}$ (°C)')
                if plot_hist:
                    line.remove()
        else:
            if 'grouplabels' in data:
                litems = []
            for i in range(xs.shape[0]):
                if not plot_hist:
                    txs = xs[i][np.where(xs[i] != None)]
                    tys = ys[i][np.where(ys[i] != None)]
                    if type(yerror[i]) == np.ndarray:
                        tyerror = yerror[i][np.where(yerror[i] != None)]
                        if len(tyerror) == 0:
                            tyerror = None
                    else:
                        tyerror = yerror[i]
                    if type(xerror[i]) == np.ndarray:
                        txerror = xerror[i][np.where(xerror[i] != None)]
                        if len(txerror) == 0:
                            txerror = None
                    else:
                        txerror = xerror[i]
                    if 'grouplabels' in data:
                        litems += [ax.errorbar(txs, tys, yerr = tyerror, xerr = txerror, elinewidth = 1, lw = ls[i], marker = m[i], capsize = 3, color = col[i], linestyle = linestyle[i])]
                    else:
                        markers, caps, bars = ax.errorbar(txs, tys, yerr = tyerror, xerr = txerror, elinewidth = 1, lw = ls[i], marker = m[i], capsize = 3, color = col[i], linestyle = linestyle[i], barsabove = False)
                else:
                    ax.hist(ys[i], orientation = 'horizontal', color = col[i])
        if 'ylabel' in data:
            ax.set_ylabel(data['ylabel'])
        if 'xlabel' in data:
            ax.set_xlabel(data['xlabel'])
        if 'labels' in data:
            plt.legend(data['labels'], fontsize = 8)
        if 'grouplabels' in data:
            itdex = data['grouplabels'][0]
            #require that itdex is of some format [(0, 1, 2), (3, 4, 5)] e.g
            for i in range(len(itdex)):
                tlist = list(itdex[i])
                for j in range(len(tlist)):
                    tlist[j] = litems[itdex[i][j]]
                itdex[i] = tuple(tlist)
            ax.legend(itdex, data['grouplabels'][1], numpoints = 1, handler_map = {tuple: HandlerTuple(ndivide = None)})
        if 'title' in data:
            plt.title(data['title'])
        if 'image' in data:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            plt.colorbar(im, cax = cax)
        if 'xax' in data:
            ax.xaxis.set_visible(data['xax'])
        if 'yax' in data:
            ax.yaxis.set_visible(data['yax'])
        if 'xsc' in data:
            ax.set_xlim(data['xsc'])
        if 'ysc' in data:
            ax.set_ylim(data['ysc'])
        if 'xlog' in data:
            ax.set_xscale('log', base = 10)
        if 'ylog' in data:
            ax.set_yscale('log', base = 10)
        if 'verrlines' in data:
            for l in data['verrlines']:
                ax.axvline(l, linestyle = 'dashed', color = (0.4, 0.4, 0.4), zorder = 100, lw = 1)
        if 'vlines' in data:
            trans = ax.get_xaxis_transform()
            for l in data['vlines']:
                ax.axvline(l, linestyle = 'dashed', color = 'k', zorder = 100, lw = 1)
                if 'vllabels' in data:
                    ax.text(l-0.0001, 1.05, data['vllabels'][int(np.where(data['vlines'] == l)[0][0])], transform = trans, rotation = 90)
        if 'herrlines' in data:
            for l in data['herrlines']:
                ax.axhline(l, linestyle = 'dashed', color = (0.4, 0.4, 0.4), zorder = 100, lw = 1)
        if 'hlines' in data:
            trans = ax.get_xaxis_transform()
            for l in data['hlines']:
                ax.axhline(l, linestyle = 'dashed', color = 'k', zorder = 100, lw = 1)
                if 'hllabels' in data:
                    ax.text(l-0.0001, 1.05, data['hllabels'][int(np.where(data['hlines'] == l)[0][0])], transform = trans, rotation = 90)
            
            
def enforce_shape(item):
    maxm = 0
    for i in range(len(item)):
        if len(item[i]) > maxm:
            maxm = len(item[i])
    for i in range(len(item)):
        if len(item[i]) < maxm:
            item[i] = np.append(item[i] , [None]*(maxm - len(item[i])))
    item = np.array(item)
    return item
    
def plot_df(vals, consts, err = [0, 0] , temp = 0, comp = 0, pixscale = [1, 'pixels'], overlay = 0):
    comps = ['raw', 'NTD', 'TM', 'noise']
    if overlay == 0:
        xplotdata = {'xs':[vals['Times']], 'ys':[vals['x_'+comps[comp]]*pixscale[0]], 'colour':[(0, 0.45, 0.7)]}
        yplotdata = {'xs':[vals['Times']], 'ys':[vals['y_'+comps[comp]]*pixscale[0]], 'xax':None, 'colour':[(0, 0.45, 0.7)]}
        if temp:
            xplotdata.update({'temps':[np.array(vals['T1s'])]})
            yplotdata.update({'temps':[np.array(vals['T1s'])]})
        if err[0]:
                xerr = np.array(vals['x_'+comps[comp]+'_err'])
                yerr = np.array(vals['y_'+comps[comp]+'_err'])
                if err[1] != 0:
                    for i in range(len(xerr)):
                        if i%int(err[1]) != 0:
                            xerr[i] = None
                            yerr[i] = None
                xplotdata.update({'yerr':[xerr]})
                yplotdata.update({'yerr':[yerr]})
    else:
        xplotdata = {'xs':[vals['Times'], vals['Times']], 'ys':[vals['x_'+comps[0]]*pixscale[0], vals['x_'+comps[comp]]*pixscale[0]], 'colour':[(0.34, 0.71, 0.91), (0, 0.45, 0.7)]}
        yplotdata = {'xs':[vals['Times'], vals['Times']], 'ys':[vals['y_'+comps[0]]*pixscale[0], vals['y_'+comps[comp]]*pixscale[0]], 'xax':None, 'colour':[(0.34, 0.71, 0.91), (0, 0.45, 0.7)]}
        if err[0]:
                xerr = np.array(vals['x_'+comps[comp]+'_err'])
                yerr = np.array(vals['y_'+comps[comp]+'_err'])
                if err[1] != 0:
                    for i in range(len(xerr)):
                        if i%int(err[1]) != 0:
                            xerr[i] = None
                            yerr[i] = None
                empty = np.zeros(len(xerr))
                for i in range(len(empty)):
                    empty[i] = None
                xplotdata.update({'yerr':[empty, xerr]})
                yplotdata.update({'yerr':[empty, yerr]})
    xplotdata.update({'xlabel':'Time (s)', 'ylabel':'Horizontal '+ f'\n displacement\n ({pixscale[1]})', 'ysc':[-0.04, 0.04]})
    yplotdata.update({'ylabel':'Vertical ' + f'\ndisplacement\n ({pixscale[1]})', 'ysc':[-0.04, 0.04]})
    plot([yplotdata, xplotdata], [[1,2], [1,2]])
    
def plot_comps(vals, consts, err = [0,0], mcomp = 4):
    cnames = ['r',r'$r_{TM}$' ,r'$r_{NTD}$', r'$r_{noise}$']
    comps = ['raw', 'NTD', 'TM', 'noise'] 
    if mcomp != 4:
        yplotdata = {'xs':[], 'ys':[], 'xax':None, 'ylabel':'Vertical '+ f'\n displacement\n (pixels)'}
        xplotdata = {'xs':[], 'ys':[], 'xlabel':'Time(s)', 'ylabel':'Horizontal '+ f'\n displacement\n (pixels)'}
        for i in range(mcomp):
            if i > 0:
                j = mcomp-i
            else:
                j = i
            yplotdata['ys'] += [vals['y_'+comps[j]]]
            yplotdata['xs'] += [vals['Times']]
            xplotdata['ys'] += [vals['x_'+comps[j+2]]]
            xplotdata['xs'] += [vals['Times']]
        xplotdata.update({'colour':[(0.34, 0.71, 0.91), (0, 0.45, 0.7), 'tab:orange']})
        yplotdata.update({'colour':[(0.34, 0.71, 0.91), (0, 0.45, 0.7), 'tab:orange'], 'labels':cnames[:mcomp]})
    else:
        for i in range(len(vals['Times'])):
            if i%250!= 0:
                vals['x_raw_err'][i] = None
                vals['x_NTD_err'][i] = None
                vals['x_TM_err'][i] = None
                vals['x_noise_err'][i] = None
        yplotdata = {'xs':[vals['Times'], vals['Times']], 'ys':[vals['x_raw'], vals['x_NTD']], 'yerr':[vals['x_raw_err'], vals['x_NTD_err']], 'xax':None, 'ylabel':'Displacement\n(pixels)', 'labels':[r'$r$', r'$r_{NTD}$']}
        xplotdata = {'xs':[vals['Times'], vals['Times']], 'ys':[vals['x_noise'], vals['x_TM']], 'yerr':[vals['x_noise_err'], vals['x_TM_err']],'xlabel':'Time(s)', 'ylabel':'Displacement\n(pixels)', 'colour':['#33BBEE', '#CC3311'], 'labels':[r'$r_{noise}$', r'$r_{TM}$']}
    plot([yplotdata, xplotdata], [[1,2], [1,2]])
    
def plot_comp_FFT(vals, consts, mcomp = 4):
    cnames = ['r',r'$r_{TM}$' ,r'$r_{NTD}$', r'$r_{noise}$']
    comps = ['raw', 'NTD', 'TM', 'noise']
    if mcomp != 4:
        yplotdata = {'xs':[], 'ys':[], 'xax':None, 'ylabel':'Power'}
        xplotdata = {'xs':[], 'ys':[], 'xlabel':'Frequency (Hz)', 'ylabel':'Power'}
        for i in range(mcomp):
            if i > 0:
                j = mcomp-i
            else:
                j = i
            dn = 10
            xFFT, xfreqs = dp.FFT(vals['x_'+comps[j]], srate = consts['s'][0])
            yFFT, yfreqs = dp.FFT(vals['y_'+comps[j]], srate = consts['s'][0])
            xFFT = abs(xFFT[1:len(vals['x_'+comps[j]])//2])
            xFFT = xFFT[:int(len(xFFT)/dn)]
            xfreqs = xfreqs[1:len(vals['x_'+comps[j]])//2]
            xfreqs = xfreqs[:int(len(xfreqs)/dn)]
            yFFT = abs(yFFT[1:len(vals['y_'+comps[j]])//2])
            yFFT = yFFT[:int(len(yFFT)/dn)]
            yfreqs = yfreqs[1:len(vals['y_'+comps[j]])//2]
            yfreqs = yfreqs[:int(len(yfreqs)/dn)]
            yplotdata['ys'] += [yFFT]
            yplotdata['xs'] += [yfreqs]
            xplotdata['ys'] += [xFFT]
            xplotdata['xs'] += [xfreqs]
        xplotdata.update({'colour':[(0.34, 0.71, 0.91), (0, 0.45, 0.7), 'tab:orange']})
        yplotdata.update({'colour':[(0.34, 0.71, 0.91), (0, 0.45, 0.7), 'tab:orange'], 'labels':cnames[:mcomp]})
    else:
        dn = 10
        rawFFT, rawfreqs = dp.FFT(vals['x_raw'], srate = consts['s'][0])
        rawFFT = abs(rawFFT[1:len(vals['x_raw'])//2])
        rawFFT = rawFFT[:int(len(rawFFT)/dn)]
        rawfreqs = rawfreqs[1:len(vals['x_raw'])//2]
        rawfreqs = rawfreqs[:int(len(rawfreqs)/dn)]
        NTDFFT, NTDfreqs = dp.FFT(vals['x_NTD'], srate = consts['s'][0])
        NTDFFT = abs(NTDFFT[1:len(vals['x_NTD'])//2])
        NTDFFT = NTDFFT[:int(len(NTDFFT)/dn)]
        NTDfreqs = NTDfreqs[1:len(vals['x_NTD'])//2]
        NTDfreqs = NTDfreqs[:int(len(NTDfreqs)/dn)]
        TMFFT, TMfreqs = dp.FFT(vals['x_TM'], srate = consts['s'][0])
        TMFFT = abs(TMFFT[1:len(vals['x_TM'])//2])
        TMFFT = TMFFT[:int(len(TMFFT)/dn)]
        TMfreqs = TMfreqs[1:len(vals['x_TM'])//2]
        TMfreqs = TMfreqs[:int(len(TMfreqs)/dn)]
        noiseFFT, noisefreqs = dp.FFT(vals['x_noise'], srate = consts['s'][0])
        noiseFFT = abs(noiseFFT[1:len(vals['x_noise'])//2])
        noiseFFT = noiseFFT[:int(len(noiseFFT)/dn)]
        noisefreqs = noisefreqs[1:len(vals['x_noise'])//2]
        noisefreqs = noisefreqs[:int(len(noisefreqs)/dn)]
        yplotdata = {'xs':[rawfreqs], 'ys':[rawFFT], 'xax':None, 'ylabel':'Power', 'labels':[r'$r$']}
        xplotdata = {'xs':[noisefreqs, TMfreqs, NTDfreqs], 'ys':[noiseFFT, TMFFT, NTDFFT],'xlabel':'Frequency (Hz)', 'ylabel':'Power', 'colour':['#33BBEE', '#CC3311', 'tab:orange'], 'labels':[r'$r_{noise}$', r'$r_{TM}$', r'$r_{NTD}$']}
        yplotdata.update({'vlines':np.array([consts['x f_TM'], consts['x f_CO']]), 'vllabels':[r'$f_{TM}$', r'$f_{CO}$'], 'ysc':[0, 30]})
        xplotdata.update({'vlines':np.array([consts['x f_TM'], consts['x f_CO']]), 'ysc':[0, 30]})
    plot([yplotdata, xplotdata], [[1,2], [1,2]])

def plot_FFT(vals, consts, err = [0, 0], comp = 0, overlay = 0):
    dn = 10
    comps = ['raw', 'NTD', 'TM', 'noise']
    xFFT, xfreqs = dp.FFT(vals['x_'+comps[comp]], srate = consts['s'][0])
    yFFT, yfreqs = dp.FFT(vals['y_'+comps[comp]], srate = consts['s'][0])
    xFFT = abs(xFFT[1:len(vals['x_'+comps[comp]])//2])
    xFFT = xFFT[:int(len(xFFT)/dn)]
    xfreqs = xfreqs[1:len(vals['x_'+comps[comp]])//2]
    xfreqs = xfreqs[:int(len(xfreqs)/dn)]
    print(xfreqs[0])
    yFFT = abs(yFFT[1:len(vals['y_'+comps[comp]])//2])
    yFFT = yFFT[:int(len(yFFT)/dn)]
    yfreqs = yfreqs[1:len(vals['y_'+comps[comp]])//2]
    yfreqs = yfreqs[:int(len(yfreqs)/dn)]
    if overlay == 1:
        bgxFFT, bgxfreqs = dp.FFT(vals['x_'+comps[0]], srate = consts['s'][0])
        bgyFFT, bgyfreqs = dp.FFT(vals['y_'+comps[0]], srate = consts['s'][0])
        bgxFFT = abs(bgxFFT[1:len(vals['x_'+comps[0]])//2])
        bgxFFT = bgxFFT[:int(len(bgxFFT)/dn)]
        bgxfreqs = bgxfreqs[1:len(vals['x_'+comps[0]])//2]
        bgxfreqs = bgxfreqs[:int(len(bgxfreqs)/dn)]
        bgyFFT = abs(bgyFFT[1:len(vals['y_'+comps[0]])//2])
        bgyFFT = bgyFFT[:int(len(bgyFFT)/dn)]
        bgyfreqs = bgyfreqs[1:len(vals['y_'+comps[0]])//2]
        bgyfreqs = bgyfreqs[:int(len(bgyfreqs)/dn)]
        yplotdata = {'xs':[bgyfreqs, yfreqs], 'ys':[bgyFFT, yFFT], 'ysc':[0, 30], 'xsc':[0, np.max(yfreqs)], 'colour':[(0.34, 0.71, 0.91), (0, 0.45, 0.7)]}
        xplotdata = {'xs':[bgxfreqs, xfreqs], 'ys':[bgxFFT, xFFT], 'ysc':[0, 30], 'xsc':[0, np.max(yfreqs)], 'colour':[(0.34, 0.71, 0.91), (0, 0.45, 0.7)]}
    else:
        yplotdata = {'xs':[yfreqs], 'ys':[yFFT], 'ysc':[0, 30], 'colour':[(0, 0.45, 0.7)]}
        xplotdata = {'xs':[xfreqs], 'ys':[xFFT], 'ysc':[0, 30], 'colour':[(0, 0.45, 0.7)]}
    yplotdata.update({'xax':None, 'ylabel':'Fourier power\nspectrum of \n vertical\ndisplacement', 'vlines':np.array([consts['y f_TM'], consts['y f_CO']]), 'vllabels':[r'$f_{TM}$', r'$f_{CO}$']})
    xplotdata.update({'xlabel':'Frequency (Hz)', 'ylabel':'Fourier power\nspectrum of \n horizontal\ndisplacement', 'vlines':np.array([consts['x f_TM'], consts['x f_CO']]), 'vllabels':[r'$f_{TM}$', r'$f_{CO}$']})
    if err[0] == 1:
        xFFT_err, xfreqs_err = dp.FFT(vals['x_'+comps[comp]+'_err'], srate = consts['s'][0])
        yFFT_err, yfreqs_err = dp.FFT(vals['y_'+comps[comp]+'_err'], srate = consts['s'][0])
        xFFT_err = abs(xFFT_err[1:len(vals['x_'+comps[comp]+'_err'])//2])
        xFFT_err = xFFT_err[:int(len(xFFT_err)/dn)]
        xfreqs_err = xfreqs_err[1:len(vals['x_'+comps[comp]+'_err'])//2]
        xfreqs_err = xfreqs_err[:int(len(xfreqs_err)/dn)]
        yFFT_err = abs(yFFT_err[1:len(vals['y_'+comps[comp]+'_err'])//2])
        yFFT_err = yFFT_err[:int(len(yFFT_err)/dn)]
        yfreqs_err = yfreqs_err[1:len(vals['y_'+comps[comp]+'_err'])//2]
        yfreqs_err = yfreqs_err[:int(len(yfreqs_err)/dn)]
        if err[1] != 0:
            for i in range(len(xFFT_err)):
                if i%err[1] != 0:
                    xFFT_err[i] = None
                    yFFT_err[i] = None
                    xfreqs_err[i] = None
                    yfreqs_err[i] = None
        empty = np.zeros(len(xFFT_err))
        for i in range(len(empty)):
            empty[i] = None
        if overlay == 1:
            yplotdata.update({'yerr':[empty, yFFT_err], 'verrlines':np.array([consts['y f_TM']-consts['y f_TM error'], consts['y f_TM']+consts['y f_TM error'], consts['y f_CO']-consts['y f_CO error'], consts['y f_CO']+consts['y f_CO error']])})
            xplotdata.update({'yerr':[empty, xFFT_err], 'verrlines':np.array([consts['x f_TM']-consts['x f_TM error'], consts['x f_TM']+consts['x f_TM error'], consts['x f_CO']-consts['x f_CO error'], consts['x f_CO']+consts['x f_CO error']])})
        else:
            yplotdata.update({'yerr':[np.array(yFFT_err)], 'verrlines':np.array([consts['y f_TM']-consts['y f_TM error'], consts['y f_TM']+consts['y f_TM error'], consts['y f_CO']-consts['y f_CO error'], consts['y f_CO']+consts['y f_CO error']])})
            xplotdata.update({'yerr':[np.array(xFFT_err)], 'verrlines':np.array([consts['x f_TM']-consts['x f_TM error'], consts['x f_TM']+consts['x f_TM error'], consts['x f_CO']-consts['x f_CO error'], consts['x f_CO']+consts['x f_CO error']])})
    plot([yplotdata, xplotdata], [[1,2], [1,2]])
    
def plot_Temps(vals, consts, err = [0,0], comp = 1, pixscale = [1, 'pixels'], pnames = [], rad = 0):
    if 'T1s' not in vals and 'T2s' not in vals:
        print('Temperatures are not available in the specified data set')
        return
    else:
        comps = ['raw', 'NTD', 'TM', 'noise']
        posdata = {'xs':[vals['Times'], vals['Times']], 'ys':[vals['y_'+comps[comp]]*pixscale[0], vals['x_'+comps[comp]]*pixscale[0]], 'labels':[f'Vertical {comps[comp]}', f'Horizontal {comps[comp]}'], 'xax':False, 'ylabel':f'Displacement\n({pixscale[1]})'}
        tempdata = {'xs':[vals['Times'], vals['Times']], 'ys':[vals['T1s']-np.average(vals['T1s']), vals['T2s']-np.average(vals['T2s'])], 'labels':['Probe 1', 'Probe 2'], 'xlabel':'Time (s)', 'ylabel':'Temperature\ndeviation (°C)'}
        if rad:
            radial = np.sqrt((np.array(posdata['ys'][0])+1)**2+(np.array(posdata['ys'][1])+1)**2)-np.sqrt(2)
            posdata['xs'] += [vals['Times']]
            posdata['ys'] += [radial]
            posdata['labels'] += [f'Radial {comps[comp]}']
        if pnames != []:
            tempdata['labels'] = pnames
    tempdata.update({'colour':['#33BBEE', '#CC3311']})
    plot([posdata, tempdata], [[1,2], [1,2]])
    
def plot_SE(vals, segsize, err = 0, pixscale = [1, 'pixels'], temps = 0):
    if temps:
        temps = [np.average(np.array(vals[0]['T1s'])), np.average(np.array(vals[1]['T1s']))]
        T1s = np.zeros([2, len(vals[0]['y_TM'])])
        for i in range(len(T1s[0])):
            T1s[0, i] = temps[0]
            T1s[1, i] = temps[1]
        yplotdata = {'xs':[np.array(vals[0]['Times'])-vals[0]['Times'][0], np.array(vals[1]['Times'])-vals[1]['Times'][1]], 'ys':[vals[0]['y_TM'], vals[1]['y_TM']], 'temps':T1s, 'hidecb':1}
        xplotdata = {'xs':[np.array(vals[0]['Times'])-vals[0]['Times'][0], np.array(vals[1]['Times'])-vals[1]['Times'][1]], 'ys':[vals[0]['x_TM'], vals[1]['x_TM']], 'temps':T1s, 'hidecb':1}
        yplotdata.update({'xax':False, 'ylabel':f'Vertical TM\n displacement\n({pixscale[1]})', 'labels':[r'$j=1$ ('+str(round(temps[0], 2))+'°C)', r'$j=N-1$ ('+str(round(temps[1], 2))+'°C)']})
        xplotdata.update({'ylabel':f'Horizontal TM\n displacement\n({pixscale[1]})', 'xlabel':'Time (s)'})
        yhistdat = {'xax':False, 'yax':False, 'method':'hist', 'xs':[np.array(vals[0]['Times'])-vals[0]['Times'][0], np.array(vals[1]['Times'])-vals[1]['Times'][1]], 'ys':[vals[0]['y_TM'], vals[1]['y_TM']], 'temps':T1s}
        xhistdat = {'xlabel':'Frequency of displacement', 'yax':False, 'method':'hist', 'xs':[np.array(vals[0]['Times'])-vals[0]['Times'][0], np.array(vals[1]['Times'])-vals[1]['Times'][1]], 'ys':[vals[0]['x_TM'], vals[1]['x_TM']], 'temps':T1s}
    else:
        yplotdata = {'xs': [np.array(vals[0]['Times'])-vals[0]['Times'][0], np.array(vals[1]['Times']) - vals[1]['Times'][1]], 'ys': [vals[0]['y_TM'], vals[1]['y_TM']]}
        xplotdata = {'xs': [np.array(vals[0]['Times'])-vals[0]['Times'][0], np.array(vals[1]['Times']) - vals[1]['Times'][1]], 'ys': [vals[0]['x_TM'], vals[1]['x_TM']]}
        yplotdata.update({'xax':False, 'ylabel':f'Vertical TM\n displacement\n({pixscale[1]})', 'labels':[r'$j=1$ ('+str(round(temps[0], 2))+'°C)', r'$j=N-1$ ('+str(round(temps[1], 2))+'°C)']})
        xplotdata.update({'ylabel':f'Horizontal TM\n displacement\n({pixscale[1]})', 'xlabel':'Time (s)'})
        yhistdat = {'xax':False, 'yax':False, 'method':'hist', 'xs':[np.array(vals[0]['Times'])-vals[0]['Times'][0], np.array(vals[1]['Times'])-vals[1]['Times'][1]], 'ys':[vals[0]['y_TM'], vals[1]['y_TM']]}
        xhistdat = {'xlabel':'Frequency', 'yax':False, 'method':'hist', 'xs':[np.array(vals[0]['Times'])-vals[0]['Times'][0], np.array(vals[1]['Times'])-vals[1]['Times'][1]], 'ys':[vals[0]['x_TM'], vals[1]['x_TM']]}
    if err[0]:
        xplotdata.update({'yerr':[]})
        yplotdata.update({'yerr':[]})
        for j in range(len(yplotdata['ys'])):    
            xerr = np.array(vals[j]['x_TM_err'])
            yerr = np.array(vals[j]['y_TM_err'])
            if err[1] != 0:
                for i in range(len(xerr)):
                    if i%int(err[1]) != 0:
                        xerr[i] = None
                        yerr[i] = None
            xplotdata['yerr'] += [xerr]
            yplotdata['yerr'] += [yerr]
    plot([yplotdata, xplotdata, yhistdat, xhistdat], [[1,2], [1,2], [1,1], [1,1]])

def plot_STD(vals, consts, CapNames, err = 0, pixscale = [1, 'pixels', 0], colours = [], linestyle = []):
    radfit = vals['radSTD'].copy()
    yfit = vals['ySTD'].copy()
    xfit = vals['xSTD'].copy()
    for i in range(len(vals['radSTD'])):
        if pixscale[2]!=0:
            vals['radSTD'][i] = np.array(vals['radSTD'][i])*pixscale[0]
            vals['radSTDerr'][i] = np.array(vals['radSTDerr'][i])*pixscale[0]+np.array(vals['radSTD'][i])/pixscale[0]*(pixscale[0]+pixscale[2])-np.array(vals['radSTD'][i])
            vals['ySTD'][i] = np.array(vals['ySTD'][i])*pixscale[0]
            vals['ySTDerr'][i] = np.array(vals['ySTDerr'][i])*pixscale[0]+np.array(vals['ySTD'][i])/pixscale[0]*(pixscale[0]+pixscale[2])-np.array(vals['ySTD'][i])
            vals['xSTD'][i] = np.array(vals['xSTD'][i])*pixscale[0]
            vals['xSTDerr'][i] = np.array(vals['xSTDerr'][i])*pixscale[0]+np.array(vals['xSTD'][i])/pixscale[0]*(pixscale[0]+pixscale[2])-np.array(vals['xSTD'][i])
        else:
            vals['radSTD'][i] = np.array(vals['radSTD'][i])*pixscale[0]
            vals['radSTDerr'][i] = np.array(vals['radSTDerr'][i])*pixscale[0]
            vals['ySTD'][i] = np.array(vals['ySTD'][i])*pixscale[0]
            vals['ySTDerr'][i] = np.array(vals['ySTDerr'][i])*pixscale[0]
            vals['xSTD'][i] = np.array(vals['xSTD'][i])*pixscale[0]
            vals['xSTDerr'][i] = np.array(vals['xSTDerr'][i])*pixscale[0]
        #radfit[i] = np.array(consts['r_exp_a'][i][0]*np.exp(np.array(vals['Temps'][i])*consts['r_exp_b'][i][0])+consts['r_exp_c'][i][0])*pixscale[0]
        #yfit[i] = np.array(consts['y_exp_a'][i][0]*np.exp(np.array(vals['Temps'][i])*consts['y_exp_b'][i][0])+consts['y_exp_c'][i][0])*pixscale[0]
        #xfit[i] = np.array(consts['x_exp_a'][i][0]*np.exp(np.array(vals['Temps'][i])*consts['x_exp_b'][i][0])+consts['x_exp_c'][i][0])*pixscale[0]
        radfit[i] = np.array(consts['rad_lin_m'][i][0]*np.array(vals['Temps'][i])+consts['rad_lin_c'][i][0])*pixscale[0]
        yfit[i] = np.array(consts['y_lin_m'][i][0]*np.array(vals['Temps'][i])+consts['y_lin_c'][i][0])*pixscale[0]
        xfit[i] = np.array(consts['x_lin_m'][i][0]*np.array(vals['Temps'][i])+consts['x_lin_c'][i][0])*pixscale[0]
    radplotdata = {'xs':vals['Temps']*2, 'ys':vals['radSTD']+radfit, 'labels':CapNames, 'ylabel':f'Radial $\sigma$ ({pixscale[1]})', 'xlabel':'Average Temperature (°C)'}
    yplotdata = {'xs':vals['Temps']*2, 'ys':vals['ySTD']+yfit, 'xax':False, 'ylabel':f'Vertical $\sigma$ \n({pixscale[1]})'}
    xplotdata = {'xs':vals['Temps']*2, 'ys':vals['xSTD']+xfit, 'ylabel':f'Horizontal $\sigma$ \n({pixscale[1]})', 'xlabel':'Avg. Temp. (°C)'}
    if err == 1:
        radplotdata.update({'yerr':vals['radSTDerr']+[[None], [None], [None]]})
        yplotdata.update({'yerr':vals['ySTDerr']+[[None], [None], [None]]})
        xplotdata.update({'yerr':vals['xSTDerr']+[[None], [None], [None]]})
    if colours != []:
        #print(colours*2)
        radplotdata.update({'colour':colours*2})
        yplotdata.update({'colour':colours*2})
        xplotdata.update({'colour':colours*2})
    if linestyle != []:
        radplotdata.update({'linestyle':linestyle})
        yplotdata.update({'linestyle':linestyle})
        xplotdata.update({'linestyle':linestyle})
    plot([radplotdata, yplotdata, xplotdata], [[2,2], [1,1], [1,1]])
    
def plot_multidf(vals, consts, cols = [], comp = 1, err = [0, 0], pixscale = [1, 'pixels'], labels = [], grouplabels = []):
    comps = ['raw', 'NTD', 'TM', 'noise']
    vertdat = {'xs':[], 'ys':[], 'xax':False, 'ylabel':'Vertical ' + comps[comp] + f'\ndisplacement \n({pixscale[1]})'}
    horizdat = {'xs':[], 'ys':[], 'ylabel':'Horizontal '+comps[comp] + f'\n displacement \n({pixscale[1]})', 'xlabel':'Time (s)'}
    for i in range(len(vals)):
        vertdat['xs'] += [vals[i]['Times']]
        vertdat['ys'] += [vals[i]['y_'+comps[comp]]*pixscale[0]]
        horizdat['xs'] += [vals[i]['Times']]
        horizdat['ys'] += [vals[i]['x_'+comps[comp]]*pixscale[0]]
    if cols != [] and len(cols) == len(vals):
        vertdat.update({'colour':cols})
        horizdat.update({'colour':cols})
    if grouplabels == []:
        vertdat.update({'labels':labels})
    else:
        vertdat.update({'grouplabels':grouplabels})
    plot([vertdat, horizdat], [[1,2], [1,2]])
    
def plot_CC(vals, crosscorr, rad=0):
    if not rad:
        bdat = {'xs':[vals['Times'], vals['Times']], 'ys':[vals['y_raw'], vals['x_raw']], 'xlabel':'Time (s)', 'ylabel':'Displacement (pixels)', 'labels':['Vertical', 'Horizontal']}
    else:
        rads = np.sqrt((np.array(vals['y_raw'])-vals['y_raw'][0])**2+(np.array(vals['x_raw'])-vals['x_raw'][0])**2)
        bdat = {'xs':[vals['Times']], 'ys':[rads], 'xlabel':'Time (s)', 'ylabel':r'$r_{rad}$ (pixels)'}
    tdat = {'xs':[vals['Times']], 'ys':[crosscorr], 'ylabel':r'$R_{ZNCC}$', 'xax':False}
    plot([tdat, bdat], [[1,2],[1,2]])