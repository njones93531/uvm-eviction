##Benchmarks

#default
- Contains applications that all run with default behavior
- Use this directory to collect access/fault patterns
- Pattern logs are stored in each application's subdirectory

#strategied
- Contains applications that are altered to allow user to specify
    access policy at runtime 
- Use this directory to conduct policy experiments
- Each application has a batch wrapper script called 'run' to 
    standardize executables. Use ./run <args>
- strategied/common/plot contains visualization scripts for 
    policy experiment data

#scripts
- Contains scripts to run experiments with either default or 
    policied behavior 
