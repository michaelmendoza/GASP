def GASP_Alpha_Sensitvity_Anaylsis():
    def analyze(D = None):
        import os
        from pathlib import Path
        import numpy as np
        from matplotlib import cm, ticker
        import matplotlib.pyplot as plt
        import gasp
        from gasp import responses
        
        width = 256
        height = 1

        gradient = 2 * np.pi
        method = 'affine'
        phantom_type='line'

        alpha0 = 10; alpha1 = 90; N = 50
        Alpha = np.linspace(alpha0, alpha1, N)
        npcs = 16
        TRs = [5e-3, 10e-3, 20e-3]
        P = [ { 'npcs':npcs, 'TRs':TRs, 'alpha': alpha } for alpha in Alpha ]
        ncoeff = npcs * len(TRs) + 1

        if D is None:
            D = responses.gaussian(width=256, bw=0.2, shift=0.-2)
        N = len(P)
        M = []
        A = np.zeros((N, ncoeff), dtype=np.complex128)
        MSE = np.zeros((N, N))

        # Train
        print(f'Training data: alpha:{alpha0}-{alpha1}, N:{N}')
        for i, p in enumerate(P):
            M1 = gasp.simulation.simulate_ssfp(width = width, height = height, 
                                                npcs = p['npcs'], TRs = p['TRs'], 
                                                alpha = np.deg2rad(p['alpha']), gradient = 2 * np.pi, 
                                                phantom_type=phantom_type, phantom_padding=0)[0] 
            Ic_train, An = gasp.train_gasp(M1, D, method=method)
            M.append(M1)
            A[i,:] = An

        print('Generating MSE')
        for i, p in enumerate(P): # Train
            for j, data in enumerate(P): # Prediction
                I = gasp.run_gasp(M[j], A[i,:], method=method)
                yhat = np.abs(I[:, 0])
                y = D
                mse = ((y - yhat)**2).mean(axis=0)  
                MSE[i,j] = mse
            

        print(f'Avg MSE: {np.mean(MSE)}')

        # Plot results
        plt.figure(figsize=(20,6))
        x = np.linspace(alpha0, alpha1, N)
        y = np.linspace(alpha0, alpha1, N)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, MSE, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
        cbar = fig.colorbar(cs)
        plt.show()

    return analyze


def GASP_Alpha_HeatMap_Butterworth(analyze):
    from gasp import responses
    D = responses.bandpass(type='butterworth', width=256, bw=0.2, shift=0.-2)
    analyze(D)