#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuerror.cuh>

/*
 * compile: nvcc -o mem_info.exe mem_info.cu -I. -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lm
 */
int main()
{
    size_t mem_free, mem_tot;
    cuErrCheck( cudaMemGetInfo( &mem_free, &mem_tot) );
    
    std::cout << "Available device memory: " << mem_free << "/" << mem_tot << " B\t" ;
    std::cout << "(" << (100. * mem_free)/((float) mem_tot) << "%)" << std::endl;
    
    
    std::cout << "Used device memory:      " << (mem_tot - mem_free) << "/" << mem_tot << " B\t" ;
    std::cout << "(" << (100. * (mem_tot - mem_free))/((float) mem_tot) << "%)" << std::endl;
    
    return EXIT_SUCCESS;
}