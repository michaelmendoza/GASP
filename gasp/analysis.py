import numpy as np
import matplotlib.pyplot as plt
from gasp import responses, simulation, gasp as GASP

def dixon_3pt(M):
    ''' 3pt dixon - M data with shape [height, width, coils, TEs]'''
    M0 = M[:,:,:,0]
    M1 = M[:,:,:,1]
    M2 = M[:,:,:,2]

    field_map = np.angle(np.conjugate(M0) * M2)
    f = np.exp(-1j * field_map / 2)
    Mw = M0 + M1
    Mf = M0 - M1
    Mw3 = M0 + M1 * np.exp(-1j * field_map / 2)
    Mf3 = M0 - M1 * np.exp(-1j * field_map / 2)

    # 2-pt dixon
    Mw = np.sqrt(np.sum(np.abs(Mw)**2, axis=2))
    Mf = np.sqrt(np.sum(np.abs(Mf)**2, axis=2))

    # 3-pt dixon
    Mw3 = np.sqrt(np.sum(np.abs(Mw3)**2, axis=2))
    Mf3 = np.sqrt(np.sum(np.abs(Mf3)**2, axis=2))

    return Mw3, Mf3

def plot_dixon(Mw, Mf):
    f = plt.figure(figsize=(8,8))
    ax = f.add_subplot(1, 2, 1)
    ax2 = f.add_subplot(1, 2, 2)
    ax.axis('off')
    ax2.axis('off')
    ax.imshow(np.abs(Mw), cmap='gray')
    ax2.imshow(np.abs(Mf), cmap='gray')
    plt.plot()

def plot_dixon_inputdata(M):
    print(M.shape)
    absM = np.sqrt(np.sum(np.abs(M)**2, axis=2))
    f = plt.figure(figsize=(8, 6))
    f.subplots_adjust(wspace=0, hspace=0)

    ax = [f.add_subplot(2,3,i+1) for i in range(6)]
    for a in ax:
        a.axis('off')
        a.set_aspect('equal')

    ax[0].imshow(np.abs(absM[:,:,0]), cmap='gray')
    ax[1].imshow(np.abs(absM[:,:,1]), cmap='gray')
    ax[2].imshow(np.abs(absM[:,:,2]), cmap='gray')
    ax[3].imshow(np.angle(M[:,:,0,0]), cmap='gray')
    ax[4].imshow(np.angle(M[:,:,0,1]), cmap='gray')
    ax[5].imshow(np.angle(M[:,:,0,2]), cmap='gray')
    plt.plot()

def gasp_train(alpha, bw, shift, method="linear"):
    width = 256
    height = 1
    npcs = 16
    TRs = [5e-3, 10e-3, 20e-3]
    alpha = np.deg2rad(alpha)
    gradient = 2 * np.pi
    phantom_type = 'line'
    
    D = responses.gaussian(width, bw=bw, shift=shift)
    M = simulation.simulate_ssfp(width=width, height=height, npcs=npcs, TRs=TRs, alpha=alpha, gradient=gradient, phantom_type=phantom_type, phantom_padding=16)
    Ic, An = simulation.train_gasp(M, D, clines=2, method=method)
    An = np.array(An)[0]
    return Ic, An, M

def gasp_plot_train(alpha = 20, bw = 0.2, shift = 0.01, method="linear"):
    Ic, An, Mtrain = gasp_train(alpha=alpha, bw=bw, shift=shift, method=method)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,2))

    _ = np.sqrt(np.sum(np.abs(Mtrain)**2, axis=2))
    _ = abs(_[:,:,0,0])
    print(_.shape)
    ax1.plot(_.T)

    _ = np.abs(Ic)
    print(_.shape)
    ax2.plot(_.T)

def gasp_run_model(Mdata, An, method="linear"):
    Ic = []
    for cc in range(Mdata.shape[2]):
        Mc = Mdata[:,:,cc,:,:]
        Mc = np.reshape(Mc,(Mc.shape[0], Mc.shape[1], -1))
        _ = GASP.apply_gasp(Mc, An, method=method)
        Ic.append(_)
    Ic = np.array(Ic)
    Ic = np.sqrt(np.sum(np.abs(Ic)**2, axis=0))
    return Ic
    
def gasp_train_and_run(Mdata, **options):
    defaultOptions = { 'method':'linear', 'alpha':20, 'shift':20, 'bw':0.2 }
    options = { **defaultOptions, **options }
    
    Ic, An, _ = gasp_train(alpha=options['alpha'], bw=options['bw'], shift=options['shift'], method=options['method'])
    output = gasp_run_model(Mdata, An, method=options['method'])

    plot_gasp_sweep([[output, Ic]])
    return output, Ic, An

def gasp_sweep(Mdata, sweep_type = 'shift', sweep_start=-0.5, sweep_end=0.5, sweep_size=10, plot_type = 'gasp', **options):
    ''' Plots a sweep of gasp images using sweep_type. You can sweep alpha, shift and bw. '''
    
    defaultOptions = { 'method':'linear', 'alpha':20, 'shift':20, 'bw':0.2 }
    options = { **defaultOptions, **options }

    dataset = []
    sweep_values = np.linspace(sweep_start, sweep_end, sweep_size)
    for val in sweep_values:
        alpha = val if sweep_type == 'alpha' else options['alpha']
        bw = val if sweep_type == 'bw' else options['bw']
        shift = val if sweep_type == 'shift' else options['shift']

        Ic, An, _ = gasp_train(alpha=alpha, bw=bw, shift=shift, method=options['method'])
        output = gasp_run_model(Mdata, An, method=options['method'])
        dataset.append([output, Ic])

    if (plot_type == 'all'):
        plot_gasp_sweep_with_training(dataset)
    if (plot_type == 'gasp'):
        plot_gasp_sweep(dataset)

def plot_gasp_sweep(dataset):
    length = len(dataset)
    f = plt.figure(figsize=(20,6))
    for i in range(length):
        data0 = dataset[i][0]
        ax = f.add_subplot(1, length, i + 1)
        ax.imshow(data0, cmap='gray')
        ax.axis('off')
    plt.show()

def plot_gasp_sweep_with_training(dataset):
    length = len(dataset)
    f = plt.figure(figsize=(20,6))
    for i in range(length):
        data0 = dataset[i][0]
        data1 = dataset[i][1]
        ax = f.add_subplot(2, length, i + 1)
        ax2 = f.add_subplot(2, length, length + i+1)
        ax.imshow(data0, cmap='gray')
        ax.axis('off')
        ax2.plot(np.abs(data1).T)
        ax2.axis('off')
    plt.show()