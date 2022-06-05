all:
	gcc -o serial serial.c -lm
	gcc -o openmp openmp.c -fopenmp -lm
	mpicc mpi.c -o mpi -lm
	nvcc -arch=sm_70 mpi.cu -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi -o mpi_cu -lm