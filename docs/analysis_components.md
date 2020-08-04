# History Matching: analysis components

This is a quick document that outlines the components and sub-systems which make up a history matching run. All of the components are customisable and can be swapped out with other code modules or left as defaults. Below is a list of references to class names / code files that is currently in use for development in approximately the order they are used:

 - Designer: converts a set of variable definitions (which define the input space) into a set of samples
 - Simulator: converts a set of input samples into a set of output samples (this is the 'real' function which is often complex and/or long-running such that querying it is avoided unless necessary).
 - Emulator: represents a statistical mapping between inputs and outputs which is (usually) cheaper to run than the main simulator
 - Diagnostic: operates on emulator and data to verify that its performance is adequate
 - (Implausibility) Transform: quantifies the likelihood of each input sample being a potential candidate for generating the reference output
 - Classifier: describes the output space in terms of regions where sets of non-implausible solutions may be found

Each of the 6 components is used in a history matching run to successively generate samples from the joint posterior distribution of the input variables which are likely to reconstruct the reference output.