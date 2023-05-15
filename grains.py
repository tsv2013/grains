import numpy as np
from skimage.draw import line
import random
import scipy.stats as st
import matplotlib.pyplot as plt

def get_grains(pixels):
    peaks = pixels.copy()
    for i in range(3, len(pixels)-3):
        if not pixels[i] and pixels[i-1] and not pixels[i-2] and pixels[i+1] and not pixels[i+2]:
            peaks[i] = True
        else:
            if pixels[i] and not pixels[i-1] and pixels[i-2] and not pixels[i+1] and pixels[i+2]:
                peaks[i] = False
    peaks = np.flip(peaks)
    grains = []
    curr_len = 0
    for i in range(len(peaks)):
        if peaks[i]:
            if curr_len > 0:
                grains.append(curr_len)
            curr_len = 0
        else:
            curr_len += 1
    return grains

def collect_grains(image):
    cuts_illustration = np.zeros(image.shape, np.uint8)
    cuts_illustration = image.copy()
    grains = []
    desired_size = 1000
    grain_border_width = 5
    pseudorandomizer = np.random.RandomState(2021)
    while len(grains) < desired_size:
        rand_r1 = random.randrange(image.shape[0])
        rand_c1 = random.randrange(image.shape[1])
        rand_r2 = random.randrange(image.shape[0])
        rand_c2 = random.randrange(image.shape[1])
        if pseudorandomizer.randint(0, 1000) < 500:
            rr, cc = line(rand_r1, 0, rand_r2, image.shape[1]-1)
        else:
            rr, cc = line(0, rand_c1, image.shape[0]-1, rand_c2)
        curr_line_pixels = image[rr, cc]
        cuts_illustration[rr, cc] = 1
        grain_candidates = get_grains(curr_line_pixels)
        for grain in grain_candidates:
            if grain > grain_border_width:
                grains.append(grain)
    return grains, cuts_illustration

def get_best_distribution(data):
    dist_names = ["norm", "lognorm", "alpha", "gamma", "beta", "expon", "exponnorm", "f", "erlang", "t", "weibull_min", "weibull_max"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

def show_best_fit(data, best_dist, best_params):
    npdata = np.array(data)
    xs = np.arange(npdata.min(), npdata.max(), 1)
    bins = npdata.max() + 1

    dist = getattr(st, best_dist)
    fit = dist.pdf(xs, *best_params)
    plt.plot(xs, fit, label=best_dist, lw=3)
    plt.hist(data, bins, density=True, label='Actual Data')
    plt.legend()
    plt.show()

def best_fit_moments(best_dist, best_params):
    dist = getattr(st, best_dist)
    (m, v, s, k) = dist.stats(*best_params, moments='mvsk')
    print('Mean and variance, calculated from best fit distribution parameters:')
    print(m)
    print(v)
