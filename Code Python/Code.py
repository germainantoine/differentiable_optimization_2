import matplotlib.pyplot as plt
import numpy as np
import pylops

def load_image_option_I(file_name = "dog_rgb.npy"):
    sampling = 5
    im = np.load(file_name)[::sampling, ::sampling, 2]
    Nz, Nx = im.shape

    # Blurring Gaussian operator
    nh = [19,21]
    hz = np.exp(-0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(-0.3 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    him = ax.imshow(h)
    ax.set_title("Blurring operator")
    fig.colorbar(him, ax=ax)
    ax.axis("tight")
    plt.show()
    Cop = pylops.signalprocessing.Convolve2D(
        (Nz, Nx), h=h, offset=(nh[0] // 2, nh[1] // 2), dtype="float32"
    )

    imblur = Cop * im
    plt.imshow(im, cmap="viridis", vmin=0, vmax=255)
    plt.show()
    plt.imshow(imblur, cmap="viridis", vmin=0, vmax=255)
    plt.show()

    Wop = pylops.signalprocessing.DWT2D((Nz, Nx), wavelet="haar", level=3)

    # This is your A and b for your f1 cost!
    A = Cop * Wop.H
    b = imblur.ravel()

    return Wop, A, b, im, imblur

def load_image_option_II(file_name = "chateau.npy"):
    sampling = 2
    im = np.load(file_name)[::sampling, ::sampling, 1]
    Nz, Nx = im.shape

    # Blurring Gaussian operator
    nh = [15, 25]
    hz = np.exp(-0.1 * np.linspace(-(nh[0] // 2), nh[0] // 2, nh[0]) ** 2)
    hx = np.exp(-0.3 * np.linspace(-(nh[1] // 2), nh[1] // 2, nh[1]) ** 2)
    hz /= np.trapz(hz)  # normalize the integral to 1
    hx /= np.trapz(hx)  # normalize the integral to 1
    h = hz[:, np.newaxis] * hx[np.newaxis, :]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    him = ax.imshow(h)
    ax.set_title("Blurring operator")
    fig.colorbar(him, ax=ax)
    ax.axis("tight")
    plt.show()
    Cop = pylops.signalprocessing.Convolve2D(
        (Nz, Nx), h=h, offset=(nh[0] // 2, nh[1] // 2), dtype="float32"
    )

    imblur = Cop * im
    plt.imshow(im, cmap="gray", vmin=0, vmax=255)
    plt.show()
    plt.imshow(imblur, cmap="gray", vmin=0, vmax=255)
    plt.show()

    Wop = pylops.signalprocessing.DWT2D((Nz, Nx), wavelet="haar", level=3)

    # This is your A and b for your f1 cost!
    A = Cop * Wop.H
    b = imblur.ravel()

    return Wop, A, b, im, imblur


def my_fista(A, b, opt_cost, eps=0.1, niter=100, tol=1e-10, acceleration=False):
    """ Here you can code your ISTA and FISTA algorithm
        Return: optimal x, and opt_gap_cost (history of cost-optcost)
    """
    #ISTA and FISTA algorithm implementation.
    # Initialization
    print(opt_cost)
    m,n = A.shape
    print(m)
    print(n)
    btaille=b.size
    #print(btaille)
    x = np.zeros(n)
    lamda = 1
    z = np.zeros(n)
    L = 2  # Placeholder for Lipschitz constant
    tau = 1/L
    opt_gap_cost = []
    if acceleration:
        for i in range(niter):
            #tau = 1/(i+1)
            x_old = x.copy()
            # Gradient step
            grad = 2*(A.T)@(A@z - b)
            y = z - tau * grad
            for j in range(x.size):
                x[j]=np.sign(y[j])*np.abs((np.abs(y[j])-tau*eps))
            #Méthode FISTA
            lamda_new = (1 + np.sqrt(1 + 4 * (lamda ** 2))) / 2
            z = ((lamda - 1 + lamda_new) / lamda_new)*x + ((1-lamda) / lamda_new) *x_old
            lamda = lamda_new
            fk = 0.5 * (np.linalg.norm(A@z - b)) ** 2 + eps * np.linalg.norm(z,1)
            opt_gap_cost.append(np.abs(fk - opt_cost))
    else:
        for i in range(niter):
            #tau = 1/(i+1)
            x_old = x.copy()
            # Gradient step
            grad = 2*(A.T)@(A@z - b)
            y = z - tau * grad
            for j in range(x.size):
                x[j]=np.sign(y[j])*(np.abs(y[j])-tau*eps)
            # Store the cost gap
            fk = 0.5 * (np.linalg.norm(A@x - b)) ** 2 + eps * np.linalg.norm(x,1)
            opt_gap_cost.append(np.abs(fk - opt_cost))
            z=x
    return x, opt_gap_cost

def run_program(A, b, Wop, eps_value=0.1, baseline_iter=1000, my_iter=100):

    # Baseline from pylops
    imdeblurfista0, n_eff_iter, cost_history = pylops.optimization.sparsity.fista(
        A, b, eps=eps_value, niter=baseline_iter
    )

    opt_cost = cost_history[-1]

    # ISTA
    my_imdeblurfista, opt_gap_cost = my_fista(
       A, b, opt_cost, eps=eps_value, niter=my_iter, acceleration=False)

    # FISTA
    my_imdeblurfista1, opt_gap_cost1 = my_fista(
       A, b, opt_cost, eps=eps_value, niter=my_iter, acceleration=True)

    #epsilon
    # my_imdeblurfista, opt_gap_costeps1 = my_fista(
    #     A, b, opt_cost, eps=1, niter=my_iter, acceleration=True)
    # my_imdeblurfista1, opt_gap_costeps2 = my_fista(
    #     A, b, opt_cost, eps=0.1, niter=my_iter, acceleration=True)
    # my_imdeblurfista, opt_gap_costeps3 = my_fista(
    #     A, b, opt_cost, eps=0.01, niter=my_iter, acceleration=True)
    # my_imdeblurfista, opt_gap_costeps4 = my_fista(
    #     A, b, opt_cost, eps=0.001, niter=my_iter, acceleration=True)

    # plt.loglog(opt_gap_costeps1, 'C0', label='epsilon = 1')
    # plt.loglog(opt_gap_costeps2, 'C1', label='epsilon = 0.1')
    # plt.loglog(opt_gap_costeps3, 'C2', label='epsilon = 0.01')
    # plt.loglog(opt_gap_costeps4, 'C3', label='epsilon = 0.01')
    plt.loglog(opt_gap_cost, 'C0', label='ISTA')
    plt.loglog(opt_gap_cost1, 'C1', label='FISTA')
    plt.grid()
    plt.loglog([3, 30], [1e6, 1e5], 'C0--', label='1/k')
    plt.loglog([3, 30], [.5e5, .5e3], 'C1--', label='1/k2')

    plt.legend()
    plt.show()

    imdeblurfista = my_imdeblurfista1.reshape(A.dims)
    imdeblurfista = Wop.H * imdeblurfista

    return imdeblurfista

def visualise_results(im, imblur, imdeblurfista):
    #Change viridis into gray for castle image.

    plt.imshow(imdeblurfista, cmap="viridis", vmin=0, vmax=250)
    plt.show()

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Deblurring", fontsize=14, fontweight="bold", y=0.95)
    ax1 = plt.subplot2grid((2, 5), (0, 0))
    ax2 = plt.subplot2grid((2, 5), (0, 1))
    ax3 = plt.subplot2grid((2, 5), (0, 2))

    ax1.imshow(im, cmap="viridis", vmin=0, vmax=250)
    ax1.axis("tight")
    ax1.set_title("Original")
    ax2.imshow(imblur, cmap="viridis", vmin=0, vmax=250)
    ax2.axis("tight")
    ax2.set_title("Blurred")

    ax3.imshow(imdeblurfista, cmap="viridis", vmin=0, vmax=250)
    ax3.axis("tight")
    ax3.set_title("FISTA deblurred")

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    plt.show()

## Load the image according to your option
Wop, A, b, im, imblur = load_image_option_I()
plt.figure()

## Run program you have coded:
imdeblurfista = run_program(A,b, Wop)

## Visualise your image results
visualise_results(im, imblur, imdeblurfista)





