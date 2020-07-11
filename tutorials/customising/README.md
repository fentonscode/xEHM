WIP: This will be a tutorial that shows how to write custom code to swap in your own analysis components.

Coming soon:

- Writing a custom emulator function (default is a Gaussian Process with a squared-exponential kernel)
- Writing a custom diagnostic function (defaults are none, constraints, leave-n-out with strict and fuzzy variants)
- Writing a custom designer function (defaults are currently full factorial and *extremely* simple Latin hypercubes)
- Writing a custom implausibility metric (default is currently based around the Mahalanobis distance with uncertainty)
- Writing a custom sampling routine (default is a cascading pseudo-rejection sampler or parallel MCMC methods)