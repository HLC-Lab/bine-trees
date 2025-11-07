> [!IMPORTANT]  
> This repository is still a work in progress. Some implementations are already provided, and others will be in the next few weeks.
> In the meanwhile, you can refer to https://github.com/HLC-Lab/pico/tree/main/libpico

# Introduction
This repository contains reference MPI implementations of the Bine Trees algorithms for collective operations.

The Bine tree algorithm is described in the paper [*"Bine Trees: Enhancing Collective Operations by Optimizing Communication Locality"*](https://arxiv.org/abs/2508.17311) by Daniele De Sensi, Saverio Pasqualoni, Lorenzo Piarulli, Tommaso Bonato, Seydou Ba, Matteo Turisini, Jens Domke, Torsten Hoefler, to be presented at SC25.

This repository is structured as follows:
- `src/`: contains the source code for the Bine Trees algorithms.
- `src_cartesian/`: contains the source code for the Bine Trees algorithms optimized for Cartesian topologies (e.g., multidimensional torus networks).
- `include/`: contains the header files for the Bine Trees algorithms.
- `Makefile`: a Makefile to build the libraries.
- `lib/`: contains the compiled libraries for the Bine Trees algorithms.
- `bin/`: contains the test scripts and executables to run the Bine Trees algorithms.
- `test/`: contains the test code for the Bine Trees algorithms.

# Building the Libraries
To build the libraries, run the following command in the root directory of the repository:
```bash
make
```
This will compile the source code and create the libraries in the `lib/` directory and the tests executable in the `bin/` directory.

# Running the Tests
To run the tests, you can run:
```bash
make test
```
This script will run the tests for all the collectives implemented in the Bine Trees algorithms. It will print the results of the tests to the console.


# Related Repositories
This repository is intended to be as simple and self-contained as possible. Other repositories are related to Bine Trees and cover
specific aspects. Namely:
- [PICO](https://github.com/HLC-Lab/pico) is a framework for benchmarking collective operations, and allows comparing Bine Trees with other algorithms available in Open MPI, Cray MPICH, and NCCL.
- [bine_trees_fugaku](https://github.com/HLC-Lab/bine_trees_fugaku) contains a port of the Bine Trees algorithms to the Fugaku supercomputer, which uses the Fujitsu A64FX ARM processor and the Tofu interconnect. This repository contains a uTofu-based implementation of the algorithms we here show in the `src_cartesian` directory.

# TODOs
- Implement non-power of two support in `src/bcast.c` similarly to standard binomial trees.
