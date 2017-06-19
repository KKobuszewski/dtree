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
 

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>

#include <math.h>
#include <complex.h>
#include <stdint.h>

#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h> 

/*
 * NOTE: Cannot be included in .cpp file, because requires implicit kernel compilation
 */

#ifndef __COMPARE_THRUST_CUH__
#define __COMPARE_THRUST_CUH__

template<typename T>
struct abs_difference : public thrust::binary_function<T,T,T>
{
  __host__ __device__
  T operator()(T x, T y) { return abs(x - y); }
};

template<typename T, int eps_exponent>
struct is_same : public thrust::binary_function<T,T,bool>
{
  __host__ __device__
  bool operator()(T x, T y) { return abs(x - y) > pow(2.0,eps_exponent); }
};

template <typename T, int eps_exponent>
inline int thrust_compare_arrays(T* d_array1, T* d_array2, const size_t size)
{
    thrust::plus<int> op_plus;
    is_same<T,eps_exponent> op_is_same;
    
    return thrust::inner_product(thrust::device_pointer_cast<T>(d_array1),thrust::device_pointer_cast<T>(d_array1)+size,
                                 thrust::device_pointer_cast<T>(d_array2), 0, op_plus, op_is_same);
}

template <typename T>
inline T thrust_total_difference(T* d_array1, T* d_array2, const size_t size)
{
    thrust::plus<T> op_plus;
    abs_difference<T> op_minus;
    
    return thrust::inner_product(thrust::device_pointer_cast<T>(d_array1),thrust::device_pointer_cast<T>(d_array1)+size,
                                 thrust::device_pointer_cast<T>(d_array2), 0, op_plus, op_minus);
}

template<typename T>
inline T thrust_max_difference(T* d_array1, T* d_array2, const size_t size, size_t* position)
{
    abs_difference<T> op_minus;
    thrust::device_vector<T> d_dest(size);
    
    thrust::transform(thrust::device_pointer_cast<T>(d_array1),thrust::device_pointer_cast<T>(d_array1)+size,
                      thrust::device_pointer_cast<T>(d_array2), d_dest.begin(), op_minus);
    
    printf("\n");
    T max_val = *(thrust::max_element( d_dest.begin(), d_dest.end() )) ;
    
    //if (!position) { *position = iter - d_dest.begin(); }
    // ;
    //std::cout << "The maximum value is " << max_val << " at position " << position << std::endl;
    return max_val;
}

template <> int thrust_max_difference<int>(int* d_array1, int* d_array2, const size_t size, size_t* position);
template <> unsigned thrust_max_difference<unsigned>(unsigned* d_array1, unsigned* d_array2, const size_t size, size_t* position);
template <> size_t thrust_max_difference<size_t>(size_t* d_array1, size_t* d_array2, const size_t size, size_t* position);
template <> float thrust_max_difference<float>(float* d_array1, float* d_array2, const size_t size, size_t* position);
template <> double thrust_max_difference<double>(double* d_array1, double* d_array2, const size_t size, size_t* position);


#endif