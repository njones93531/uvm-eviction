# Nick Progress

## Reproduce
 - Successful on Voltron

### More benchmarks
 - Added spmm-csr
    - Currently has a managed and explicit version
       - Using explicit version for testing with shims
    - Takes in Gbs as an argument (as well as density and CPU check)
 - Added Centrality
    - Dr. Saules previous work (graph processing)
       - Has been updated to current nvcc version (by Dr. Saule)
    - Needs more testing
       - Arguments for runs differ from other benchmarks
          - Use graphs as input (would require wget setup or something)
 - Added basic knapsack
    - Some code I wrote a while ago, could be interesting to look at

## Automate collect logs
 - auto\_run.sbatch can be run like this:
    - sbatch auto\_run.sbatch \<managed/normal\>
 - auto\_logs.py is used to:
    - insmod logging driver and run logging script
    - and exec (some hardcoded stuff to be changed in future)
 - auto\_conv.sh is optional (auto\_run.sbatch managed)
    - copies benchmark to new directory before changing files
    - this script attempts to change an explicitly managed cuda file to a UVM managed file
       - h* variables declarations and mallocs are destroyed
       - d* variables are changed to h* variables
       - cudaMallocs to cudaMallocManged
       - etc.
    - at this point this option is not very strong, some assumptions made
       - therefore, if the make fails after this conversion we suggest the user to make manual changes

## Automate collect metrics
 - have adjusted auto\_logs.py to correctly name and place logging data
 - use existing metric\_plot.py in order to dump metrics file
 - TODO: make decisions about how to pick policies

## Automate implementing policy
 - benchmarks/strategied/auto\_test/ contains a test space for this
 - currently conv.sh in that dir works on basic testing files
   - called for preview:
      - bash conv.sh input.cu <string of policies (hhd)>  
   - to file
      - bash conv.sh input.cu <string of policies (hhd)> > output.cu
 - TODO: needs more testing for complex files
