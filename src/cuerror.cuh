/* *********************************************************************** *
 *   WARSAW UNIVERSITY OF TECHNOLOGY                                       *
 *   FACULTY OF PHYSICS                                                    *
 *   NUCLEAR THEORY GROUP                                                  *
 *                                                                         *
 *   Author: Konrad Kobuszewski                                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 *                                                                         *
 * *********************************************************************** */ 
 

#ifndef CUERROR_CUH_
#define CUERROR_CUH_


#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex.h>      // not std::complex!
#include <cuComplex.h>
#include <cufft.h>
#include <cublas_v2.h>





/* ************************************************************************************ *
 *                                                                                      *
 *                                GPE ERRORS HANDLING                                   *
 *                                                                                      *
 * ************************************************************************************ */


// =================================== CUDA ERROR HANDLING ===================================================

/*
 * This macro enables simple handling of cudaError_t, and passes error as gpe_result_t to gpe_exec macro
 * TODO: if could be in gpe_engine.cuh?
 */                                                                            \

inline void _cuErrCheck(const cudaError_t err, const char *file, const int line)
{
  if(err != cudaSuccess)
      {
          fprintf( stderr, "\n");
          fprintf( stderr, "CUDA ERROR in file=`%s`, line=%d\n",file,line);
          fprintf( stderr, "CUDA ERROR %d: %s\n", err, cudaGetErrorString((cudaError_t)(err)) );
          fprintf( stderr, "\n");
          cudaDeviceReset();
          exit(EXIT_FAILURE);
      }
}

#define cuErrCheck(err)         _cuErrCheck(err, __FILE__, __LINE__)


// =================================== CUFFT ERROR HANDLING ==================================================

static const char *cufftGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "CUFFT_INCOMPLETE_PARAMETER_LIST";
        case CUFFT_INVALID_DEVICE:
            return "CUFFT_INVALID_DEVICE";
        case CUFFT_NO_WORKSPACE:
            return "CUFFT_NO_WORKSPACE";
        case CUFFT_PARSE_ERROR:
            return "CUFFT_PARSE_ERROR";
        case CUFFT_NOT_IMPLEMENTED:
            return "CUFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";
        default:
        	return "UNKNOWN ERROR!";
    }
}

inline void _cufftErrChk(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
        fprintf( stderr, "\n");
        fprintf( stderr, "CUFFT ERROR in file=`%s`, line=%d\n",file,line);
        fprintf( stderr, "CUFFT ERROR %d: %s\n", err, cufftGetErrorEnum((cufftResult)(err)) );
        fprintf( stderr, "\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

#define cufftErrChk(err)      _cufftErrChk(err, __FILE__, __LINE__)


// =================================== CUBLAS ERROR HANDLING =================================================

/* 
 * This function enables simple handling of cublasStatus_t (status of cublas-library operation)
 * 
 * DEVELOPE IT: if there is error, it should show what the error is !!!
 */
static inline void CheckCublas( cublasStatus_t status,
                         const char *file,
                         int line ) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf( "CUBLAS error in %s at line %d\n", file, line );
        exit( EXIT_FAILURE );
    }
} 

#define CHECK_CUBLAS( status ) (CheckCublas(status, __FILE__, __LINE__)) 


#endif /* CUERROR_CUH_ */
