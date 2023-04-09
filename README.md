# High-order (FFT) based solver for simulating non-isothermal binary dendrite
This is a Direct Numerical Simulation (DNS) based phase field implementation of a binary alloy solidification problem, simulating the microscopic dendrite growth.
# Software requirements
This solver needs:

- g++/gcc
- FFTW3
# How to complie and run the code

To compile the code

```bash
g++ kr_2dbinary_dendrite_nonconserv_filtering_ssprk3.cpp -lfftw3 -o a.out
```
To run this code

```bash
./a.out ./ run1
```
