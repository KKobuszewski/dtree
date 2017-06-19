#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import pycuda.driver as cuda
import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
import string





cuda_source = r"""

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define N          ($LEN_DATA) 
#define NUM_NODES  ($NUM_NODES)
#define NUM_LEVELS ($NUM_LEVELS)

//#define DEBUG

typedef $REAL_TYPE real_t;

/*
 * TODO: Think of using Texture Memory!!!
 */
// condition
__constant__ real_t  x_cond[NUM_NODES];       // stores conditions to check for each node 
__constant__ uint8_t point_idx[NUM_NODES];    // indicates which element of data point to use

// binary tree structure
__constant__ int     left_child[NUM_NODES];   // indicates index of left child of current node
// TODO: Think if can replace by int16_t ???


inline __device__ bool check_condition(real_t x, real_t a)
{
    return (x > a);
}


/*
 *  @param real_t*   data             - Poiter to array NxNUM_LEVELS containing data to classify
 *  @param uint64_t* classification   - 
 */
__global__ void kernel_dtree_test(real_t *data, uint64_t* classification, const int data_depth)
{
    // int tid = threadIdx.x + blockDim.x*blockIdx.x;
    int tid = threadIdx.x; // ACHTUNG! Apply only one block!!!
    
    #ifdef DEBUG
    if (tid == 0)
    {
        printf("Nodes:       %u \n",NUM_NODES);
        printf("Levels:      %u \n",NUM_LEVELS);
        printf("Len data:    %u \n",N);
        printf("data dep:    %d \n",data_depth);
        printf("real_t:      %u \n",sizeof(real_t));
        printf("shared mem:  %u \n",NUM_LEVELS*N*sizeof(real_t));
    }
    #endif
    
    
    // ============== load data =====================================================
    
    // Shared memory for data read.
    // NOTE: Limits num of threads per block.
    // 65536 = NUM_LEVELS*N * sizeof(real_t) !!!
    __shared__ real_t __data_shared[NUM_LEVELS*N*sizeof(real_t)]; // be aware of memory banks, TODO!!! 
    
    for (uint8_t lvl=0; lvl < data_depth; lvl++)
    {
        // TODO: Apply some transformations here!
        __data_shared[lvl*N+tid] = data[lvl*N+tid];
    }
    
#ifdef DEBUG
    printf("%u\t[%3.3f,%3.3f,%3.3f]\n",tid,__data_shared[0*N+tid],__data_shared[1*N+tid],__data_shared[2*N+tid]);
    if (tid==0) printf("\n\n");
#endif
    
    
    // ============== decision tree traversal ===========================================
    
    uint64_t result           = 0;                   // 64 bits of storage space <- IMPLIES THAT MAXIMALLY WE HAVE 64 LEVELS
    int      current_node_idx = 0;
    uint8_t  __point_idx;
    
    // TODO: Unroll this loop!
    for (uint8_t lvl=0; lvl < NUM_LEVELS; lvl++)     // NOTE: maximally 256 levels! (now ok, because storing results in 64 bits)
    {
        // find which element of data point to investigate
        __point_idx = point_idx[current_node_idx];
        
        // TODO:
        // Reorder data to get rid of idle threads!
        
        // check condition
        bool condition = check_condition(__data_shared[__point_idx*N+tid],x_cond[current_node_idx]);
        
        // set bit on lvl-th position
        result |= ((uint64_t) condition)<<lvl; // NOTE: result variable should have only zero bits at the beginning  
                                                // TODO: if casting and bit shift will generate what we want? <- CHECK
        
        // find next node index
        
#ifdef DEBUG
      printf("%u.\t%d\t%u\t%d\t%lu\n",tid,current_node_idx,condition,left_child[current_node_idx] + condition,result);  
#endif
        current_node_idx = left_child[current_node_idx] + condition;
        if (current_node_idx < 0) {break;} // end iteration (NOTE: -1 means we are in the leaf node!)
    }
    
    
    
    // ============== store results of classification =====================================================
    
    // TODO: assign class here using result variable
    classification[tid] = result;
    
#ifdef DEBUG
    printf("%u.\t[%3.3f,%3.3f,%3.3f]\t %lu \n",tid,data[0*N+tid],data[1*N+tid],data[2*N+tid],result);
#endif
}

"""








# how to copy to constant mem
# d_constant = = module.get_global('d_constant')[0]
# cuda.memcpy_htod(d_constant,  x) # x is numpy array of compatible type


"""
DTREE checking where the variable lies
 structure:          level
 0                       0
 |        \               
 1         2             1
 |  \      |  \           
 3  4      5  6          2
 |  |  \   |  |  \        
-1  7  8  -1  9  10      3

 0  0  0   1  1  1   binary output representation
 0  1  1   0  1  1
 0  0  1   0  0  1
 0  -  -   0  -  -
 
 0  2  6   1  3  9   decimal output representation
    10 14     11 15
"""




if __name__ == '__main__':
    # define sturcture of the tree
    dtree_struct   = [[ 1],                           # describes next node to be used
                      [ 3, 5],
                      [-1, 7,-1, 9],
                      [-1,-1,-1,-1]]
    dtree_x_idx    = [[ 0],                           # describes feature that node uses for classification
                      [ 1, 1],
                      [ 2, 2, 2, 2],
                      [ 2, 2, 2, 2]]
    dtree_x_cond   = [[ 0.0],                         # condition a that node evaluates (x > a)
                      [ 0.0, 0.0],
                      [ 0.0, 0.0, 0.0, 0.0],
                      [-1.0,-1.0, 1.0, 1.0]]
    list_flatten   = lambda l,dtp: np.array([item for sublist in l for item in sublist],dtype=dtp)
    dtree_flatten  = list_flatten(dtree_struct,np.int32)
    num_nodes      = dtree_flatten.size
    num_levels     = 3+1
    real_type      = 'float'
    real_type_size = 4
    
    print('------------------------------------------------------------------------------')
    print('DTree')
    print()
    print('levels:     ',num_levels)
    print('nodes:      ',num_nodes)
    print('mem. size:  ',num_nodes*(real_type_size + 4 + 1),'B')
    print('max nodes:  ',65536/(real_type_size + 4 + 1), '(const mem limitation)')
    print('max threads:',65536/num_nodes/num_levels/real_type_size, '(shared mem limitation)')
    print()
    print('structure:  ')
    for t in dtree_struct:
        print(t)
    print()
    print('------------------------------------------------------------------------------')
    print()
    print()
    
    
    # generate data to classify
    N = 256
    mean = np.array([1, 1, 1])
    cov  = np.array([[1,0,0], [0,0.1,0], [0,0,0.1]])
    
    y = np.random.randint(2, size=N, dtype=np.uint8)
    data = np.multiply(y,np.random.multivariate_normal(mean, cov, N).T) + \
                         np.multiply((1-y),np.random.multivariate_normal(-mean, cov, N).T)
    data = np.array(data,dtype=np.float32)
    print(data.shape)
    print(data.dtype)
    print(data.T)
    print(y[:6])
    print(data[0,N-1],data[1,0])
    print(data.flatten()[N-1:N+1])
    
    
    # compile cuda module
    template    = string.Template(cuda_source)
    cuda_source = template.substitute(LEN_DATA   = N,
                                      NUM_NODES  = num_nodes,
                                      NUM_LEVELS = num_levels,
                                      REAL_TYPE  = 'float')
    module = SourceModule(cuda_source)
    
    
    print('# copying dtree structure to device')
    d_dtree_struct = module.get_global('left_child')[0]
    d_dtree_xcond  = module.get_global('x_cond')[0]
    d_dtree_x_idx  = module.get_global('point_idx')[0]
    cuda.memcpy_htod(d_dtree_struct, dtree_flatten)
    cuda.memcpy_htod(d_dtree_xcond, list_flatten(dtree_x_cond,np.float32))
    cuda.memcpy_htod(d_dtree_x_idx, list_flatten(dtree_x_idx,np.uint8) )
    print(dtree_flatten,dtree_flatten.dtype)
    
    
    
    print('# load data to GPU')
    h_result = np.empty(N,dtype=np.uint64)
    d_data   = cuda.mem_alloc_like(data.flatten())
    d_result = cuda.mem_alloc_like(h_result)
    cuda.memcpy_htod(d_data, data.flatten())
    
    
    
    print('# perform classification on device')
    kernel = module.get_function('kernel_dtree_test')
    kernel(d_data,d_result,np.int32(data.shape[0]),block=(N,1,1),grid=(1,1,1))
    
    
    print('# copy back results to host')
    cuda.memcpy_dtoh(h_result,d_result)
    
    
    
    print('# visualize results')
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    colors = ['red','darkorange','yellowgreen','orange','green','royalblue','darkgreen','wheat',
              'navy','indigo','blue','azure','yellow','brown','grey','black']
    for it,u in enumerate(np.unique(h_result)):
        f = data[:,np.where(h_result == u)]
        ax.scatter(f[0],f[1],f[2],c=colors[it])
    plt.show()