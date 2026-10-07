# bel(x_t) = p(x_y|z_{1:t},u_{1:t})
# -----  Markov Chain assumption ------
#
# State at the present:
#	 p(X_t|x_{t-1},u_t)
# State probability at state t: 
#	p(z_t|x_t)
#


# ----- Rekursive bayes filtering defnition -----
#
# prior state = p(x_0)
# CurrentState = PastState · (observations)
#
# ------ Predicttion -----
#
#	p(x_t|z_{1:t-1})=
#
# ------ correction ------
#
# 	bel(x_t)= p(x_t|z_{1:t})
#
# Dyncamical model
# x_t ~ p(x_t|x_{t-1},u_t)
# x_t = f(x_{t-1},u_t) + w_t
#
# sthocatic differnece -> distribution of current state. 
#
# Observation model
# z_t~p(z_t|x_t)
# z_t = g(x) + v_t
#
# Distrubution of current state -> sthocastic difference. 
#
# particle samples algorithim uses weighted particle but must still som to 1. 
# 



# ----- particle filters concept ------. 
# 1. Prediction
# move past state particles according to current observations,
# then diffuse with added noise to the new particle locations.
#
# 2. Correction
# Perfom an observation, and and use this to compute weigths
# for the samp
# 3. Resamplin
# sample particles i.i.d from the state distribution given the updated
# weights, with replacement
#
# the amount off samples taken, needs to enough to represent the 
# state space, each itteration off the algorithim nees to sample the
# same amount off samples.  noise per paticles need to be random. 


# ----- proposal distributions -----
#
# using a much simpler propozal distribution q(z) and sampling from that
# side steps the diffictuly in sampling from the original distribution
# p(z)
#
#1.
# generate iid samples from q(z), giving M samples. 
# 
#2.
# Evaluate the approximate in each sample \in M. 
# Introduce a weighting term so shift the expecatations to be
# respective to the original distribution.
#
#
# w_m = {p(z^m)}/{q(z^m)}


# ----- Sir Algorithim. 

#1. genereate iid form q(z)
#2. compute importance weights -> normalize.
#3. Resample weights based on normalized sample
# With replacement, iid with respect to weight, allowing multiple
# samples off same  particle.






