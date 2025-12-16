# FluidMaxxer

FluidMaxxer is an LBM implementation written in C++ and CUDA.  
It is mainly intended for the analysis of species transport in porous media.

The current implementation uses the BGK collision operator for Navier-Stokes (NS)
as well as advection-diffusion (AD) lattices. The lattices are coupled via a velocity
coupling approach. Linear and Langmuir isotherm adsorption models are implemented.

The code supports reading geometry files in VTI format.

Planned extensions include the implementation of more advanced collision operators
(TRT, MRT, porous media BGK collision models), improvements to the unit conversion,
and additional simulation options.

## Prerequisites

- CUDA
- C++17 compatible compiler

## Build and Run

Currently, the only example is a coupled NS–AD simulation with adsorption,
implemented in `main.cpp`.

Build instructions:

```bash
mkdir build
cd build
cmake ..
make
./lbm_cuda_ad
