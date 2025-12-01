import numpy as np

def finite_difference(y, i, dt, order=1):
    """
    Compute the finite difference of y at index i with a time step dt.
    https://web.media.mit.edu/~crtaylor/calculator.html

    params:
        y: list of vectors. [N, D]
        i: indices of the points to compute the derivative at.
        dt: float
        order: order of differentiation. 
    """
    N = len(y)

    if N < 5:
        raise ValueError("The array size must be greater than 5. Got: ", N)

    if order == 0:
        return y
    elif order == 1:
        y_x = (1*y[i-2]-8*y[i-1]+0*y[i+0]+8*y[i+1]-1*y[i+2])/(12*1.0*dt**1)
        return y_x
    elif order == 2:
        y_xx = (-1*y[i-2]+16*y[i-1]-30*y[i+0]+16*y[i+1]-1*y[i+2])/(12*1.0*dt**2)
        return y_xx
    elif order == 3:
        y_xxx = (-1*y[i-2]+2*y[i-1]+0*y[i+0]-2*y[i+1]+1*y[i+2])/(2*1.0*dt**3)
        return y_xxx
    else:
        raise ValueError("Only 0, 1, 2, 3 orders are implemented.")



# wrote a test to check the correctness of the finite_difference function by creating a sine wave and then computing the derivative of the sine wave at a point using the finite_difference function.
def test():
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    yd = np.cos(x)
    ydd = -np.sin(x)
    yddd = -np.cos(x)

    dt = x[1] - x[0]
    ids = np.arange(10, 90)
    yd_fd = finite_difference(y, ids, dt, 1)
    ydd_fd = finite_difference(y, ids, dt, 2)
    yddd_fd = finite_difference(y, ids, dt, 3)

    # make a subplot for each of yd, ydd, yddd
    fig, axs = plt.subplots(3, 1, layout='constrained')



    axs[0].plot(x[ids], yd[ids], label="yd")
    axs[0].plot(x[ids], yd_fd, label="yd_fd")
    axs[1].plot(x[ids], ydd[ids], label="ydd")
    axs[1].plot(x[ids], ydd_fd, label="ydd_fd")
    axs[2].plot(x[ids], yddd[ids], label="yddd")
    axs[2].plot(x[ids], yddd_fd, label="yddd_fd")
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    plt.show()

    


if __name__ == "__main__":
    test()