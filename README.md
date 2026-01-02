# Fouriers-AI

This project explores a novel approach to fitting and learning periodic trajectories, such as circles or other parametric curves, using concepts inspired by classical PID (Proportional-Integral-Derivative) control. Traditional curve-fitting methods often struggle with stability and long-term divergence, particularly when optimizing multi-rotation trajectories.

We introduce a differentiable, PD-inspired loss function that combines:

Proportional (P) term: measures instantaneous Euclidean distance between predicted and target points.

Derivative (D) term: captures the rate of change of error along the curve, promoting smoothness and reducing overshoot.

Optional Integral (I) term: initially explored to accumulate past error, though removed to improve long-term stability.

This project also explores other forms of error and their affects on Fourier Series based approximations. 

Quick Result Description: The PID error method given <50,000 epochs was capable of finding a set of values for each circle that could near perfectly fit an input circle of size 200 for nearly two full rotations. For > 2 rotations, the graph begins to diverge from the path assigned to it. This issue most likely arises from the training itself and not from the error function. More on this later. 


The outcome demonstrates a novel methodology for stable, differentiable trajectory fitting, leveraging control theory principles in a machine learning context. This framework is applicable to any periodic or parametric trajectory learning task, potentially extending to robotics, animation, and other domains requiring precise motion replication.
