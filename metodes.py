import numpy as np
from scipy.optimize import fsolve

def Euler(f, xSpan, IC, nOfSteps):
    h = (xSpan[1] - xSpan[0]) / nOfSteps
    x = np.linspace(xSpan[0], xSpan[1], nOfSteps+1)
    y = np.zeros( (len(IC), nOfSteps + 1) )
    y[:, 0] = IC
    for i in np.arange(0, nOfSteps):
        y[:, i + 1] = y[:, i] + h * f(x[i], y[:, i])
    return x, y

def EulerBackward(f, xSpan, IC, nOfSteps):
    h = (xSpan[1] - xSpan[0]) / nOfSteps
    x = np.linspace(xSpan[0], xSpan[1], nOfSteps+1)
    y = np.zeros((len(IC), nOfSteps + 1))
    y[:, 0] = IC
    for i in np.arange(0, nOfSteps):
        # Definim la funció para resoldre usant fsolve
        def backward_step(ynext):
            return y[:, i] + h * f(x[i + 1], ynext) - ynext
        
        # Resoldre usant fsolve per obtenir el següent valor de y
        y[:, i + 1] = fsolve(backward_step, y[:, i])
    return x, y


def RK2(f, xSpan, IC, nOfSteps):
    h = (xSpan[1] - xSpan[0]) / nOfSteps
    x = np.linspace(xSpan[0], xSpan[1], nOfSteps+1)
    y = np.zeros( (len(IC), nOfSteps + 1) )
    y[:, 0] = IC
    for i in np.arange(0, nOfSteps):
        k1 = f(x[i], y[:,i])
        k2 = f(x[i]+h,y[:,i]+h*k1)
        y[:, i + 1] = y[:, i] + h * (k1+k2)/2
    return x, y

def RK4(f, xSpan, IC, nOfSteps):
    h = (xSpan[1] - xSpan[0]) / nOfSteps
    x = np.linspace(xSpan[0], xSpan[1], nOfSteps+1)
    y = np.zeros( (len(IC), nOfSteps + 1) )
    y[:, 0] = IC
    for i in np.arange(0, nOfSteps):
        k1 = f(x[i], y[:,i])
        k2 = f(x[i]+h/2,y[:,i]+h*k1/2)
        k3 = f(x[i]+h/2,y[:,i]+h*k2/2)
        k4 = f(x[i]+h,y[:,i]+h*k3)
        y[:, i + 1] = y[:, i] + h*(k1 + 2*k2 + 2*k3 + k4)/6
    return x, y

def RKF45(f, xSpan, IC, nOfSteps, tol):
    h = (xSpan[1] - xSpan[0]) / nOfSteps
    x = np.linspace(xSpan[0], xSpan[1], nOfSteps+1)
    y = np.zeros( (len(IC), nOfSteps + 1) )
    y[:, 0] = IC
    for i in np.arange(0, nOfSteps):
        k1 = f(x[i], y[:,i])
        k2 = f(x[i] + h/4, y[:,i] + h*k1/4)
        k3 = f(x[i] + 3/8 * h, y[:,i] + h*(3/32 * k1 + 9/32 * k2))
        k4 = f(x[i] + 12/13 * h, y[:,i] + h*(1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
        k5 = f(x[i] + h, y[:,i] + h*(439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4))
        k6 = f(x[i] + 1/2 * h, y[:,i]+ h*(- 8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5))
        y[:,i+1] = y[:,i] + h*(25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5)
        Aprox_e = h*(1/136 * k1 - 128/4275 * k3 - 2197/75240 * k4 + 1/50  * k5 + 2/55 * k6)
        if(np.linalg.norm(Aprox_e)<tol): 
            h = h*(tol/np.linalg.norm(Aprox_e))**(1/5)
    return x, y
    