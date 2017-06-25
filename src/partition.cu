
#define CUDA_API_PER_THREAD_DEFAULT_STREAM    // one stream per cpu thread

#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>



typedef float real_t;

struct prg
{
    real_t a, b;
    
    __host__ __device__
    prg(real_t _a=0.f, real_t _b=1.f) : a(_a), b(_b) {};
    
    __host__ __device__
    real_t operator()(const unsigned int n) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<real_t> dist(a, b);
        rng.discard(n);
        
        return dist(rng);
    }
};


#define CONDITION ((real_t)1.0)

template <bool truefalse, typename T>
struct checkCondition //: public thrust::unary_function<real_t,bool>
{
    const T cond;
    
    checkCondition(T c=1.0) : cond(c) {};
    
    __host__ __device__
    bool operator()(T data) 
    {
        if (truefalse)
            return (data > cond);
        else
            return (data < cond);
    }
};

//void make_partition( real_t* d_data, const unsigned n_points, const unsigned m_elements );


template <typename T>
void make_partition( thrust::device_vector<T>& numbers, 
                     thrust::device_vector<T>& result, 
                     T condition )
{
    // check condition
    /*
    thrust::transform(numbers.begin(), numbers.end(), conditions.begin(), checkCondition<true,T>(condition));
    
    const unsigned size_first = thrust::reduce(conditions.begin(), conditions.end());
    std::cout << size_first << " numbers fullfills predicate " << std::endl;
    */
    thrust::detail::normal_iterator<thrust::device_ptr<real_t>> last_copied = 
    thrust::copy_if(        numbers.begin(), numbers.end(), result.begin(), checkCondition<true,T>(condition));
    thrust::remove_copy_if( numbers.begin(), numbers.end(), last_copied, checkCondition<true,T>(condition));
    std::cout << "Elements copied to group 0: " << result.end() - last_copied    << std::endl;
    std::cout << "Elements copied to group 1: " << last_copied  - result.begin() << std::endl;
}

template <typename T>
struct is_zero : public thrust::unary_function<bool,T>
{
  __host__ __device__
  bool operator()(T s_val) { return (!s_val);}
};

template <typename T>
void make_partition_many( T* d_data, 
                          T* d_result, 
                          /*thrust::device_vector<bool>& conditions,*/
                          T condition,
                          const unsigned rows, const unsigned cols, const unsigned feature )
{
    // check condition
    // TODO: probably not needed
    /*thrust::transform(thrust::device_pointer_cast(d_data+feature*rows), 
                      thrust::device_pointer_cast(d_data+(feature+1)*rows-1), 
                      conditions.begin(), checkCondition<true,T>(condition));*/
    
    //const unsigned size_first = thrust::reduce(conditions.begin(), conditions.end());
    //std::cout << size_first << " numbers fullfills predicate " << std::endl;
    
    int size_group_0;
    
    // TODO: use openmp and make default stream to be different for each host thread
    // TODO: If for small data is also efficient?
    #pragma omp parallel for num_threads(4)
    for (unsigned it=0; it < cols; it++)
    {
        thrust::detail::normal_iterator<thrust::device_ptr<real_t>> last_copied;
        last_copied = thrust::copy_if(thrust::device_pointer_cast(d_data+it*rows),        // begining of data chunk to copy
                                      thrust::device_pointer_cast(d_data+(it+1)*rows),  // end of data chunk to copy
                                      thrust::device_pointer_cast(d_data+feature*rows),   // the feature we want to check condition
                                      thrust::device_pointer_cast(d_result+it*rows),      // where to copy
                                      checkCondition<true,T>(condition));
        thrust::remove_copy_if( thrust::device_pointer_cast(d_data+it*rows),              // begining of data
                                thrust::device_pointer_cast(d_data+(it+1)*rows),        // end of data
                                thrust::device_pointer_cast(d_data+feature*rows),         // feature
                                last_copied,                                              // iterator pointing to the end of 
                                checkCondition<true,T>(condition));
        if (omp_get_thread_num() == 0)
        {
            size_group_0 = last_copied - thrust::device_vector<real_t>::iterator(thrust::device_pointer_cast(d_result));
            std::cout << "Elements copied to group 0: " << size_group_0 << std::endl;
            std::cout << "Elements copied to group 1: " << rows - size_group_0 << std::endl;
        }
    }
}


template <typename T>
inline  int make_partition_many2( T* d_data, 
                                 T* d_result, 
                                 T condition,
                                 const unsigned rows, const unsigned cols, const unsigned feature )
{
    int size_group_0 = 0;
    // TODO: use openmp and make default stream to be different for each host thread ?
    #pragma omp parallel for num_threads(4)
    for (unsigned it=0; it < cols; it++)
    {
        thrust::detail::normal_iterator<thrust::device_ptr<real_t>> last_copied;
        last_copied = thrust::copy_if(thrust::device_pointer_cast(d_data+it*rows),        // begining of data chunk to copy
                                      thrust::device_pointer_cast(d_data+(it+1)*rows),  // end of data chunk to copy
                                      thrust::device_pointer_cast(d_data+feature*rows),   // the feature we want to check condition
                                      thrust::device_pointer_cast(d_result+it*rows),      // where to copy
                                      checkCondition<true,T>(condition));
        thrust::remove_copy_if( thrust::device_pointer_cast(d_data+it*rows),              // begining of data
                                thrust::device_pointer_cast(d_data+(it+1)*rows),        // end of data
                                thrust::device_pointer_cast(d_data+feature*rows),         // feature
                                last_copied,                                              // iterator pointing to the end of 
                                checkCondition<true,T>(condition));
    
        if (omp_get_thread_num() == 0)
           size_group_0 = last_copied - thrust::device_vector<real_t>::iterator(thrust::device_pointer_cast(d_result));
    }
    
    return size_group_0;
}


template <typename T1, typename T2>
inline  int make_partition_one ( T1* d_to_partition,
                                 T1* d_result, 
                                 T2* d_data, 
                                 T2 condition,
                                 const unsigned rows, const unsigned cols, const unsigned feature )
{
    int size_group_0 = 0;
    
    // make partition on classes and find size of groups
    typename  thrust::detail::normal_iterator<thrust::device_ptr<T1>> last_copied;
    last_copied = thrust::copy_if(thrust::device_pointer_cast(d_to_partition),        // begining of data chunk to copy
                                  thrust::device_pointer_cast(d_to_partition+rows),   // end of data chunk to copy
                                  thrust::device_pointer_cast(d_data+feature*rows),   // the feature we want to check condition
                                  thrust::device_pointer_cast(d_result),              // where to copy
                                  checkCondition<true,T2>(condition));
    size_group_0 = last_copied - 
                   typename thrust::device_vector<T1, thrust::device_malloc_allocator<T1>>::
                   iterator(thrust::device_pointer_cast(d_result));
    thrust::remove_copy_if(       thrust::device_pointer_cast(d_to_partition),        // begining of data chunk to copy
                                  thrust::device_pointer_cast(d_to_partition+rows),   // end of data chunk to copy
                                  thrust::device_pointer_cast(d_data+feature*rows),   // the feature we want to check condition
                                  last_copied,                                        // where to copy
                                  checkCondition<true,T2>(condition));
    
    
    return size_group_0;
}



template <typename T>
inline void print_thrust_vector(thrust::device_ptr<T> vec_beg,thrust::device_ptr<T> vec_end)
{
    thrust::copy(vec_beg, vec_end, std::ostream_iterator<T>(std::cout, "   "));
    std::cout << std::endl;
}

template <typename T>
inline void print_thrust_vector(thrust::device_vector<T>& vec)
{
    thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(std::cout, "   "));
    std::cout << std::endl;
}
 
/*
__global__ void kernel_rearange_array( , const array_size)
{
    
}
*/

int main()
{
    const unsigned N = pow(2,4)*400;
    
    thrust::device_vector<real_t> numbers(N);
    thrust::device_vector<real_t> result(N);
    thrust::device_vector<unsigned> conditions(N);
    //thrust::device_vector<unsigned> prescan1(N);
    //thrust::device_vector<unsigned> prescan2(N);
    
    thrust::counting_iterator<unsigned> index_sequence_begin(0);
    
    
    std::cout << "sizeof bool:  " << sizeof(bool)    << std::endl;
    std::cout << "sizeof uint8: " << sizeof(uint8_t) << std::endl;
    //std::cout << std::setw(2);
    
    // vector of random numbers
    thrust::transform(index_sequence_begin, index_sequence_begin + N, numbers.begin(), prg(0.f,2.f));
    if (N < 20) print_thrust_vector(numbers);
    
    // check condition
    thrust::transform(numbers.begin(), numbers.end(), conditions.begin(), checkCondition<true,real_t>(1.0));
    if (N < 20) print_thrust_vector(conditions);
    const unsigned size_first = thrust::reduce(conditions.begin(), conditions.end());
    std::cout << size_first << " numbers fullfills predicate " << std::endl;
    
    // make scan
    /*thrust::exclusive_scan(conditions.begin(), conditions.end(), prescan1.begin(), 0);
    thrust::copy(prescan1.begin(), prescan1.end(), std::ostream_iterator<real_t>(std::cout, "   "));
    std::cout << std::endl;*/
    
    // 
    make_partition<real_t>(numbers,result,1.0);
    if (N < 20) print_thrust_vector<real_t>(result);
    
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    
    
    thrust::transform(index_sequence_begin, index_sequence_begin + N, numbers.begin(), prg(0.f,2.f));
    if (N < 20) print_thrust_vector(numbers);
    
    if (N < 20)
    for ( unsigned it=0; it < N; it += (N/4) )
    {
        // thrust::device_pointer_cast(thrust::raw_pointer_cast(result.data())+it)
        print_thrust_vector<real_t>( thrust::device_pointer_cast<real_t>(numbers.data()+it),
                                     thrust::device_pointer_cast<real_t>(numbers.data()+it+(N/4)) );
    }
    std::cout << std::endl;
    
    make_partition_many<real_t>( thrust::raw_pointer_cast(numbers.data()),
                                 thrust::raw_pointer_cast(result.data()),
                                 1.0,
                                 N/4, 4, 0 );
    
    // now 
    if (N < 20)
    for ( unsigned it=0; it < N; it += (N/4) )
    {
        // thrust::device_pointer_cast(thrust::raw_pointer_cast(result.data())+it)
        print_thrust_vector<real_t>( thrust::device_pointer_cast<real_t>(result.data()+it),
                                     thrust::device_pointer_cast<real_t>(result.data()+it+(N/4)) );
    }
    
    
    
    
    return EXIT_SUCCESS;
}
