"this file documents the mirte groups solution to the SIR problem"
import numpy

def SIR(N: int, p, q, sampler):
    particles = sampler(N)
    wheights = p(particles)/q(particles)
    weights = weights / np.sum(weights)
    weights = np.cumsum(wheights)
    pick = numpy.random.rand()
    for i in range(len(wheights)):
        if pick >= wheights[i]:
            return particles[i]
    raise ValueError('somthing went wrong')

def()
