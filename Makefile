# Mumbai flood sim - PC project ODD-2026
# Rehan Mehta (241080039), Bhavya Gothi (241080021)
#
#   make            builds all four versions into bin/
#   make vecreport  shows which loops gcc vectorised
#   make clean

CXX    ?= g++
MPICXX ?= mpicxx

# -ffp-contract=off : no fused multiply-add, so serial and parallel answers match bit for bit
# -fopenmp-simd     : honour "omp simd" even in the serial / mpi builds
CXXFLAGS = -O3 -march=native -std=c++17 -Wall -Wextra -ffp-contract=off -fno-math-errno -fopenmp-simd
MPIFLAGS = -DUSE_MPI -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX

SRC  = src/flood.cpp
DEPS = src/flood.cpp src/params.hpp src/domain_io.hpp

all: bin/flood_serial bin/flood_omp bin/flood_mpi bin/flood_hybrid

bin:
	mkdir -p bin

bin/flood_serial: $(DEPS) | bin
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

bin/flood_omp: $(DEPS) | bin
	$(CXX) $(CXXFLAGS) -fopenmp $(SRC) -o $@

bin/flood_mpi: $(DEPS) | bin
	$(MPICXX) $(CXXFLAGS) $(MPIFLAGS) $(SRC) -o $@

bin/flood_hybrid: $(DEPS) | bin
	$(MPICXX) $(CXXFLAGS) $(MPIFLAGS) -fopenmp $(SRC) -o $@

# baseline for the SIMD experiment: libm cbrt and no vectorisation at all
bin/flood_novec: $(DEPS) | bin
	$(CXX) -O3 -march=native -std=c++17 -ffp-contract=off -fno-math-errno -fno-tree-vectorize -DSCALAR_CBRT $(SRC) -o $@

# only the non-MPI ones, for machines without MPI installed
nompi: bin/flood_serial bin/flood_omp

vecreport: | bin
	$(CXX) $(CXXFLAGS) -fopenmp -fopt-info-vec-optimized -c $(SRC) -o bin/vec.o 2>&1 | grep flood.cpp

clean:
	rm -rf bin

.PHONY: all nompi vecreport clean
