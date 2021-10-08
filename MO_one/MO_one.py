import numpy as np
import math
import tabulate
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from scipy.optimize import line_search, minimize
from typing import Callable

#(2*x+y+1)^4+0.1*(x + 2*y-1)^4
#

def dy(x):
    return 4*(x[0]**2+x[1]-11) + 2*(x[0] + x[1]**2 - 7)
    #return -np.exp(-x[0]-x[1])*(3*x[0]**2+5*x[1]**2)+6*x[0]*np.exp(-x[0]-x[1])
    #return 200 *x[1] - 400*x[0]**2

def dx(x):
    return 2*(x[0]**2+x[1]-11) + 4*(x[0] + x[1]**2 - 7)
    #return -np.exp(-x[0]-x[1])*(3*x[0]**2+5*x[1]**2)+10*x[1]*np.exp(-x[0]-x[1])
    #return 2 * (800 * x[0] ** 3 - 400*x[0]*x[1] + x[0] - 1)

def fun(x):
    return (x[0]**2+x[1]-11)**2 + (x[0] + x[1]**2 - 7)**2
    #return np.exp(-x[0]-x[1])*(3*x[0]**2+5*x[1]**2)
    #return 100 * ((x[1] - 2*(x[0]**2)) ** 2) + (1 - x[0]) ** 2

#def dy(x):
#    return 8*(2*x[0]+x[1]+1)**3 + 0.4*(x[0]+2*x[1]-1)**3

#def dx(x):
#    return 4*(2*x[0]+x[1]+1)**3 + 0.8*(x[0]+2*x[1]-1)**3

#def fun(x):
#    return (2*x[0]+x[1]+1)**4+0.1*(x[0] + 2*x[1]-1)**4

def grad(x):
    return  np.array([dy(x), dx(x)]).flatten()

def printDecorator(f, res):
    def wrapper(x):
        res.append(x)
        ret_val = f(x)
        return ret_val
    return wrapper

def trunc(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def steepest_descent(fun: Callable[[list], float], gfun:Callable[[list], list], point: tuple, eps: float, _callback:Callable[[dict], None] = None):
    def f1d(alpha):
        return fun(point + alpha*s)

    ndim = len(point)

    fun_counter = 0
    gfun_counter = 0

    while(True):
        if _callback:
            _callback({"x": point, "f": fun(point), "fnum": fun_counter, "gfnum": gfun_counter})

        s = -gfun(point)
        gfun_counter +=1
        golden_res = sopt.golden(f1d, full_output=1)
        alpha = golden_res[0]
        fun_counter += golden_res[2]


        new_point = point + np.dot(alpha, s)

        if np.abs(np.prod(point) - np.prod(new_point)) < eps:
            point = new_point
            break
 
        point = new_point
    return {"x": point, "f": fun(point), "fnum": fun_counter, "gfnum": gfun_counter}

def Broiden(fun: Callable[[list], float], gfun:Callable[[list], list], point: tuple, A: np.ndarray, eps: float, _callback:Callable[[dict], None] = None) -> dict:
    def f1d(alpha):
        return fun(point + alpha*s)

    fgrad = gfun(point)
    ndim = len(point)

    fun_counter = 0
    gfun_counter = 1

    while(True):
        if _callback:
            _callback({"x": point, "f": fun(point), "fnum": fun_counter, "gfnum": gfun_counter})

        s = np.dot(-A, fgrad)

        golden_res = sopt.golden(f1d, full_output=1)
        alpha = golden_res[0]
        fun_counter += golden_res[2]
        new_point = point + np.dot(alpha, s)

        new_fgrad = gfun(new_point)
        delta_x = np.array(new_point - point)
        delta_g = np.array(new_fgrad - fgrad)
        d = delta_x - np.dot(A, delta_g)
        A = A + np.dot(d.reshape((ndim, 1)), d.reshape((1, ndim)))/np.dot(d.reshape((1, ndim)), delta_g.reshape((ndim, 1)))
        #p = np.dot(A, delta_g)
        #A = A + (np.dot(delta_x.reshape((ndim, 1)), delta_x.reshape((1, ndim)))/np.dot(delta_x.reshape((1, ndim)), delta_g.reshape((ndim, 1)))) - (np.dot(p.reshape((ndim, 1)), p.reshape((1, ndim)))/np.dot(p.reshape((1, ndim)), delta_g.reshape((ndim, 1))))
        
        if np.abs(np.prod(point) - np.prod(new_point)) < eps:
            point = new_point
            fgrad = new_fgrad
            break
        
        point = new_point
        fgrad = new_fgrad

        gfun_counter += 1

    return {"x": point, "f": fun(point), "fnum": fun_counter, "gfnum": gfun_counter}

def plot(xlim, ylim, zlim, x1lim, y1lim):
    xs = []
    results_list = []
    f = lambda x: xs.append(x["x"])
    callback = printDecorator(f, results_list)
    eps = 1e-6
    #print(steepest_descent(fun, grad, first_point, eps, callback))
    print(Broiden(fun, grad, first_point, np.eye(len(first_point)), eps, callback))

    #Print Results Table
    header = results_list[0].keys()
    rows =  [x.values() for x in results_list[0:11] + [results_list[-1]]]
    print(tabulate.tabulate(rows, header, tablefmt='grid'))

    X, Y = np.mgrid[xlim[0]:xlim[1]:20j, ylim[0]:ylim[1]:20j]
    Z = fun(np.array([X, Y]))

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')

    ax.plot_wireframe(X, Y, Z)

    ax.set_zlim3d(zlim[0], zlim[1])

    ax.plot(np.array(xs).T[0], np.array(xs).T[1], np.array(
        list(map(fun, np.array(xs)))).T, "x-", color='red')


    ax1 = fig.add_subplot(1, 2, 2)

    X, Y = np.mgrid[x1lim[0]:x1lim[1]:30j,y1lim[0]:y1lim[1]:30j]
    Z = fun(np.array([X, Y]))

    ax1.contour(X, Y, Z, levels=40)

    t = 0
    for x in zip(np.array(xs).T[0], np.array(xs).T[1]):
        if abs(fun(x) - t) > 1e-2:
            ax1.annotate(trunc(fun(x),3), (x[0], x[1]))
        t = fun(x)
        #ax1.plot(x[0], x[1], "x-", color='red')
    ax1.plot(np.array(xs).T[0], np.array(xs).T[1], "x-", color='red')
        

    plt.show()

first_point = (1, 1)
#first_point = (0,0)

#print(Broiden(fun, grad, first_point, np.eye(len(first_point)), 1e-10))
#plot((-1.7, 1.5), (0, 2.5), (0, 500))
plot((-5, 2), (-5, 2), (0, 110), (-4, 1.5), (-3.6, 1.5))