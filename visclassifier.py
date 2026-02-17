import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.matlib import repmat

def visclassifier(fun,xTr,yTr,w=None,b=0):
    """
    visualize decision boundary
    """
    
    yTr = np.array(yTr).flatten()
    w = np.array(w).flatten() if w is not None else None

    symbols = ["ko","kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    classvals = np.unique(yTr)

    plt.figure()

    res=300
    xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]),res)
    yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]),res)
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    with torch.no_grad():
        xTe = (np.array([pixelX.flatten(), pixelY.flatten()]).T)
        if isinstance(xTr, torch.Tensor):
            testpreds = fun(torch.FloatTensor(xTe)) # .detach().numpy()
        else:
            testpreds = fun(xTe)

    if isinstance(testpreds, torch.Tensor):
        testpreds = testpreds.numpy()
        
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c,0],
            xTr[yTr == c,1],
            marker=marker_symbols[idx],
            color='k'
           )

    if w is not None:
        alpha = -1 * b / (w ** 2).sum()
        plt.quiver(w[0] * alpha, w[1] * alpha,
            w[0], w[1], linewidth=2, color=[0,1,0])

    plt.axis('tight')
    plt.show()
