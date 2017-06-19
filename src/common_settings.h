#ifndef __COMMON_H__
#define __COMMON_H__

#define CUDA_API_PER_THREAD_DEFAULT_STREAM    // one stream per cpu thread, TODO: check if it really works...

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#include <omp.h>  // TODO: Add openMP and many streams!

#include <cuda.h>
#include <cuda_runtime.h>

// ======================== common settings ===============================

#define MAX_DTREES        256
#define MAX_CLASSES       256
#define NUM_CLASSES       3
#define MAX_FEATURES      512  // maximal number of cols in dataset
#define MAX_COLS         (MAX_FEATURES)
#define NUM_SPLITS        64
#define MAX_DEPTH         64   // sizeof (size_t) - every path in the tree can be represented by 


#define MAX_CPU_THREADS 8
#define NUM_STREAMS    (MAX_CPU_THREADS*2)  // there are two copy engines on gpu, so no need for more streams?

#define WORKSPACE_ELEMENTS 1024*128
#define WORKSPACE_SIZE (WORKSPACE_ELEMENTS*sizeof(real_t))



//#define USE_DOUBLE

// define real types
#ifndef USE_DOUBLE
typedef float   real_t;
typedef float2  real2_t;
typedef float3  real3_t;
typedef float4  real4_t;
#else
typedef double   real_t;
typedef double2  real2_t;
typedef double3  real3_t;
typedef double4  real4_t;
#endif





#endif