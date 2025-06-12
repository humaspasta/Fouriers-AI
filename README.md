# Fouriers-AI

Having learning about the Fourier Series in its discrete form, I wanted to attempt a project where I explored this idea in the context of artifical intelligence. The Fourier Series is a formula that sums up a lot of wave functions in order to estimate another function. This is much like the Taylor Series except for the fact that the Taylor Series uses Polynomial instead. Series such as the Taylor Series and the Fourier Series are very helpful in the context of regression and AI and are used in many other fields of engineering and mathematics. 

My basic idea here was to take advantage of the Fourier Series and pair it with a deep learning algorithm that would use the frequencies, radii, and phase of 7 different circles in order to re-draw a given input pattern (in my case a circle of radius 200 to start). As the algorithm learns the correct values for all three paramters, the Fourier Series in its discrete form, should output the exact same pattern periodically. 

This model is not generalizable and works on a single piece of data at a time. It re-trains on this data so that it can get exact values and fit to the pattern as perfectly as it can. This may defeat the purpose of deep learning algorithms but becomes especially useful when the patten becomes more and more complicated. While there are other mathematical ways of doing something like this...I dont care. 

The next steps are to reduce error and try and fit the model to other kinds of images. This could be any  sort of polar graph or custom image that a user wants to insert. 


