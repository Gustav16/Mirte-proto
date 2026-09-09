"this file documents the mirte groups solution to the SIR problem"
import numpy as np
import math
import matplotlib.pyplot as plt

sqrt_of_2_PI = math.sqrt(2*math.pi)

def gaussian_PDF(x: float, mu: float, sigma: float) -> float:
    "gaussian PDF function"
    return (1/(sqrt_of_2_PI*sigma))*np.exp(-(1/2)*((x-mu)**2/(sigma**2)))

#make ixture of gaussian model
p = lambda x:  0.3 * gaussian_PDF(x, 2, 1) + 0.4*gaussian_PDF(x, 5, 2) + 0.3*gaussian_PDF(x, 9, 1)
q = lambda x: 1/15
sampler = lambda n: np.random.uniform(1, 15, n)

def SIR(N: int, p, q, sampler):
    """Function for Sampling-Importance-Resampling,
        takes a sampling amount N: int, a pdf p: x -> float and pdf q: x -> float and a sampler: N -> (i.i.d sample)^N as arguments"""
    particles = sampler(N)
    weights = p(particles)/q(particles)
    weights = weights /np.sum(weights)
    weights = np.cumsum(weights)

    #Get and return resampled particles tfrom the distribution
    picks = np.random.rand(N)
    indices = np.searchsorted(weights, picks) #use binsearch to get indices
    return particles[indices]
print(SIR(10, p, q, sampler)) 

#question 3.2
for N in [20, 100, 1000]:
    samples = SIR(N, p, q, sampler)
    x = np.linspace(1, 15, 1000)
    plt.figure()
    
    # density=True makes the histogram a probability density
    plt.hist(samples, bins=30, density=True, alpha=0.5, label="SIR samples")

    # Target distribution
    plt.plot(x, p(x), label="p(x)")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title(f"SIR Q1 with N = {N}")
    plt.legend()
    plt.show()



#question 3.3
q = lambda x: gaussian_PDF(x, 5, 4)
sampler = lambda n: np.random.normal(5, 4, n)

print(SIR(10, p, q, sampler)) 
for N in [20, 100, 1000]:
    samples = SIR(N, p, q, sampler)
    x = np.linspace(-10, 20, 1000)
    plt.figure()
    
    # density=True makes the histogram a probability density
    plt.hist(samples, bins=30, density=True, alpha=0.5, label="SIR samples")

    # Target distribution
    plt.plot(x, p(x), label="p(x)")

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title(f"SIR Q2 with N = {N}")
    plt.legend()
    plt.show()
