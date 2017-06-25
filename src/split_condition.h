#ifndef __SPLIT_CONDITION_H__
#define __SPLIT_CONDITION_H__

#include <DecisionTree.hpp>

#define CUB_STDERR
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>


// ============================================== MACRO DEFINITIONS ===============================================================

#define DIV_UP(x,y)   ((x + y - 1)/y)

// blocks properties
// BUG: BLOCK_SIZE = 128 gives asshole values
#define BLOCK_SIZE              192      // Number of threads per block, check 192 for this data set or 384
#define ITEMS_PER_THREAD        4         // NOTE: probably should be multiple of 4 to use real4_t for load!
#define ITEMS_PER_BLOCK        (BLOCK_SIZE*ITEMS_PER_THREAD)
#define SPLITS_PER_BLOCK        2                                  // number of splits evaluated by single block 
                                                                   // (NOTE: Equal to splits per fractions chunk, but there could be more blocks per chunk!)  


// these must be macro functions, because on every level of the tree we will have different values
// blocks grid properties
#define BLOCKS_PER_CHUNK(rows)   ( DIV_UP(rows,ITEMS_PER_BLOCK) )                           // How many blocks per single row of data per fractions chunk 
#define NUM_BLOCKS(rows)         ( BLOCKS_PER_CHUNK(rows)*(NUM_SPLITS/SPLITS_PER_BLOCK)  )                // How many blocks for row of data for every condition
#define TOT_BLOCKS(rows,cols)    ( NUM_BLOCKS(rows)*cols ) 

                                                                   
// chunk properties
#define CHUNKS_PER_ROW               ( NUM_SPLITS/SPLITS_PER_BLOCK )       // number of fractions chunk per single feature
#define NUM_CHUNKS(cols)             ( cols*CHUNKS_PER_ROW )
#define CHNUK_ID(block_id_x,rows)    ( block_id_x/BLOCKS_PER_CHUNK(rows) ) // chunk_id in row


// ==================================================


inline void print_launch_params(const unsigned rows, const unsigned cols)
{
    printf("BLOCK_SIZE:            %d\n",BLOCK_SIZE);
    printf("ITEMS_PER_THREAD:      %d\n",ITEMS_PER_THREAD);
    printf("ITEMS_PER_BLOCK:       %d\n",ITEMS_PER_BLOCK );
    printf("SPLITS_PER_BLOCK:      %d\n",SPLITS_PER_BLOCK);
    printf("                         \n");
    printf("NUM_CHUNKS:            %d/%d\n",CHUNKS_PER_ROW,NUM_CHUNKS(cols));
    printf("BLOCKS_PER_CHUNK:      %d\n",BLOCKS_PER_CHUNK(rows));
    printf("NUM_BLOCKS:            %d/%d\n",NUM_BLOCKS(rows),TOT_BLOCKS(rows,cols));
    
}




// ================================================================================================================

/*
 * 
 * @param uint8_t* classes      - pointer to array describing which class the point belongs to
 * @param int*     storage      - pointer to array for storing number of elements belonging each class in data
 * @param unsigned rows         - number of data points for trial splits
 * 
 */
__global__ void kernel_find_classes_counts_tot(uint8_t* classes, int* storage, const unsigned rows)
{
    int idx     = threadIdx.x + blockDim.x*blockIdx.x;
    
    int _counter;
    int _aggregate;
    
    // load data
    int _classes[ITEMS_PER_THREAD] = {0};
    for (int it=0; it < ITEMS_PER_THREAD; it++)
        _classes[it] = (int) classes[idx*ITEMS_PER_THREAD + it]; // TODO: Make extensible to cases where rows is not a multiple of ITEMS_PER_THREAD
    
    // clear memory
    if (idx < NUM_CLASSES) storage[idx] = 0;
    //for (int classes_it=0; classes_it < NUM_CLASSES; classes_it++) 
    //    storage[classes_it] = 0;
    
    typedef cub::BlockReduce<int, BLOCK_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    
    
    if (idx*ITEMS_PER_THREAD < rows)
    {
        for (int classes_it=0; classes_it < NUM_CLASSES; classes_it++)
        {
            //_aggregate = 0;
            _counter   = 0;
            for (int it=0; it < ITEMS_PER_THREAD; it++)
                 _counter += (_classes[it] == classes_it) && 1; // NOTE: with only == can produce strange results
            
            _aggregate = BlockReduceT(temp_storage).Sum(_counter);
            __syncthreads();
            
            if (threadIdx.x == 0)
                atomicAdd(&storage[classes_it], _aggregate);
            //__syncthreads();
        }
        
    }
    
}

/*
 * 
 * @param int*     classes      - pointer to array describing which class the point belongs to (NOTE: uint8_t casted to int!)
 * @param int*     storage      - pointer to array for storing number of elements belonging each class in data
 * @param unsigned rows         - number of data points for trial splits
 * 
 */
__global__ void kernel_find_classes_counts_tot1(int* classes, int* storage, const unsigned rows)
{
    int idx     = threadIdx.x + blockDim.x*blockIdx.x;
    
    int _counter;
    int _aggregate;
    
    // load data
    int _classes[ITEMS_PER_THREAD/4] = {0};
    uint8_t classes_casted[ITEMS_PER_THREAD];
        
    
    typedef cub::BlockReduce<int, BLOCK_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    
    
    // clear memory
    if (idx < NUM_CLASSES) storage[idx] = 0; 
    
    if (idx*ITEMS_PER_THREAD < rows)
    {
        // load data
        for (int it=0; it < ITEMS_PER_THREAD/4; it++)
            ((int*) classes_casted)[it] = (int) classes[idx*(ITEMS_PER_THREAD/4) + it];
        
        // apply for each class type
        for (int classes_it=0; classes_it < NUM_CLASSES; classes_it++)
        {
            // find number of elements belonging to current class
            for (int it=0; it < ITEMS_PER_THREAD/4; it++)
            {
                _classes[it] = 0;
                _classes[it] += (classes_casted[it*(ITEMS_PER_THREAD/4) + 0] == classes_it) && 1;
                _classes[it] += (classes_casted[it*(ITEMS_PER_THREAD/4) + 1] == classes_it) && 1;
                _classes[it] += (classes_casted[it*(ITEMS_PER_THREAD/4) + 2] == classes_it) && 1;
                _classes[it] += (classes_casted[it*(ITEMS_PER_THREAD/4) + 3] == classes_it) && 1;
            }
            
            _aggregate = BlockReduceT(temp_storage).Sum(_classes);
            __syncthreads();
            
            if (threadIdx.x == 0)
                atomicAdd(&storage[classes_it], _aggregate);
            __syncthreads();
        }
    }
    
}



// ================================================================================================================




// global gpu storage ???
//__device__ real_t fractions[NUM_SPLITS*NUM_CLASSES*MAX_FEATURES];

#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for (uint8_t offset = WARP_SIZE/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {

  static __shared__ T shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}


/*
 * @param real_t*  data         - pointer to data array
 * @param uint8_t* classes      - pointer to array describing which class the point belongs to
 * @param int*     fractions    - pointer to array for storing number of counts in group 1 for each trial split and class
 * @param int*     group_sizes  - pointer to array for storing number of elemnts in group 1 for each trial split
 * @param unsigned rows         - number of data points for trial splits
 * @param unsigned cols         - number of features of each data point
 * 
 * Shared memory is allocated statically and should be enough for BLOCK_SIZE=1024
 * 
 * To be used  <<< (NUM_BLOCKS(rows),cols,1), (BLOCK_SIZE,1,1) >>>
 * NOTE: Second dimension of grid represents features!
 * 
 * TODO: Make templated function!
 * TODO: Use template metaprograming on device functions to unroll loops!
 * 
 */
__global__ void kernel_find_fractions(real_t* data, uint8_t* classes,                    /*dataset*/
                                            int* fractions, int* group_sizes,            /*storage*/
                                            const unsigned rows, const unsigned cols)    /*dimensions*/
{
    //int  tid = threadIdx.x*ITEMS_PER_THREAD;        // thread id inside block
    const int data_id = threadIdx.x*ITEMS_PER_THREAD + 
                        ITEMS_PER_BLOCK*( blockIdx.x%BLOCKS_PER_CHUNK(rows) );                          // id of data element to be read, TODO: no modulo!!!
    
    const unsigned feature = blockIdx.y;                                                                     // second dimension of grid iterates over features! 
    unsigned       frac_id = feature*NUM_SPLITS*NUM_CLASSES +                                            // move index in dependence on which feature we process
                             NUM_CLASSES*SPLITS_PER_BLOCK*(blockIdx.x/BLOCKS_PER_CHUNK(rows));                     // add split index in row
    unsigned       split_id = SPLITS_PER_BLOCK*(blockIdx.x/BLOCKS_PER_CHUNK(rows));
    
    
    // allocate memory
    typedef cub::BlockReduce<int, BLOCK_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    
    int _conditions = 0; // NOTE: Must be int because of __shuf_down intrinsic, uint16_t and uin8_t can generate undefined behavoiur, sadly...
    uint8_t  _classes[ITEMS_PER_THREAD] = {0};
    real_t   _data[ITEMS_PER_THREAD] = {0};
    
    real_t x_min = min_in_feature[feature];
    real_t df   = (max_in_feature[feature] - x_min)/((real_t) NUM_SPLITS);
    
    if ( data_id < rows ) //if ( data_id + ITEMS_PER_THREAD <= rows )
    {
    
    // load <- done once, because every chunk and split need to be evaluated on same data
    for (uint8_t it=0; it < ITEMS_PER_THREAD; it++)
        _classes[it] = classes[ data_id + it ];
    
    for (uint8_t it=0; it < ITEMS_PER_THREAD; it++)
        _data[it]    = data[ feature*rows + data_id + it ];
    
    
    // TODO: Unroll this loops
    for (unsigned it_split = 0;   it_split   < SPLITS_PER_BLOCK; it_split++)                                 // enables evaluation of few splits inside same block
    for (unsigned it_classes = 0; it_classes < NUM_CLASSES;      it_classes++)                               // split must be evaluated for each class
    {
        /*
        frac_id = feature*NUM_SPLITS*NUM_CLASSES +                        
                   CHNUK_ID(block_id_x,rows)*SPLITS_PER_BLOCK*NUM_CLASSES +
                   it_split*NUM_CLASSES + (it_classes-1);
        */
        
        // Find condition of trial split
        // find split condition: this is 
        // adding 0.5 because edge values are hardly possible
        real_t cond = x_min + df*( CHNUK_ID(blockIdx.x,rows)*SPLITS_PER_BLOCK + it_split + 0.5 );
        // this is split_id * (rows/NUM_SPLITS), integer division sholud round down,
        // so we read always from row of data according to feature
        // NOTE: In this way we have random splits...
        //cond =  data[ feature*rows + (rows/NUM_SPLITS)*(split_id+it_split) ];
        
        
        // Evaluate thread sums for conditions
        // If elements belongs to group 1 (_data[it] > cond) and belongs to current class
        _conditions = 0;
        for (uint8_t it=0; it < ITEMS_PER_THREAD; it++)
            _conditions += (_classes[it] == it_classes) && (_data[it] > cond);
        __syncthreads(); // check if needed?
        
        // Make reduction on conditions and find how many elements belong to gruop 1
        // NOTE: there are BLOCKS_PER_CHUNK writing to same place in global mem
        // NOTE: fractions must be set to zeros before!!!
        int sum = BlockReduceT(temp_storage).Sum(_conditions); // 
        //sum = (real_t) blockReduceSum<int>(_conditions) / ((real_t) rows);
        
        __syncthreads();
        
        // TODO: check if two blocks read/write from __global__ not simultaneously !!! <- in another case this can generate undefined behaviour
        // aggregate to __global__ mem
        if (threadIdx.x == 0)
            atomicAdd(&fractions[frac_id], sum);
        
        
        // count how many items in 1 group (once per split condition)
        if (it_classes ==0)
        {
            _conditions = 0;
            for (uint8_t it=0; it < ITEMS_PER_THREAD; it++)
                _conditions += 1 && (_data[it] > cond);
            __syncthreads();
            
            int sum2 = BlockReduceT(temp_storage).Sum(_conditions); //
            __syncthreads();
            
            if (threadIdx.x == 0)
                atomicAdd(&group_sizes[NUM_SPLITS*feature + split_id + it_split], sum2);
        }
        __syncthreads();
        
        
        
#ifdef DEBUG
        //if (blockIdx.x < 32 && feature==0 && threadIdx.x == 0 && it_split == 0)
        //    printf("block: %4d  feature:%4d  frac_id:%4d   (x >%.2f) \t sum: %.0f\n", blockIdx.x, feature, frac_id, cond, sum);
        if (threadIdx.x == 0 && it_classes == 0 && (blockIdx.x%(8* BLOCKS_PER_CHUNK(rows))) ==0)
        {
            printf("%4d. (%d)\tsplit:%4d/%4d\tcondition: x > %.2f\tsum: %4d\tfraction: %4d\tgroup: %4d\n",
                   blockIdx.x,feature,
                   split_id+it_split,frac_id,
                   cond,sum,fractions[frac_id],
                   group_sizes[NUM_SPLITS*feature + split_id + it_split]  );
        }
#endif
        
        
        frac_id++;
    }
    
    } // if (data_id < rows)
}




/*
 * Function describing cost of information. Should be summed for all frac[].
 * The smaller value gives the feature, the better split provides.
 * We assume that cost function has value in range [0.0,1.0]
 * 
 * @param  const real_t frac - fraction of class occurres in a group after current split
 * @return real_t            - information cost for current split
 */
__inline__ __device__ real_t cost_function(const real_t frac)
{
    return frac*(1-frac); // now using Gini Index
}


/*
 * This should be one block??
 * 
 * Must be in another kernel to synchronize blocks.
 * 
 * 
 * @param int*     fractions    - pointer to array for storing number of counts in group 1 for each trial split and class
 * @param int*     storage      - pointer to array for storing number of elements belonging each class in data
 * @param int*     group_sizes  - pointer to array for storing number of elemnts in group 1 for each trial split
 * @param unsigned rows         - number of data points for trial splits
 * 
 */
__global__ void kernel_find_split(int* fractions, int* storage, int* group_sizes, const unsigned rows)
{
    
    const int      split_id = (threadIdx.x + blockDim.x*blockIdx.x);  // id of data element to be read
    const unsigned feature  = blockIdx.y;                             // second dimension of grid iterates over features! 
    
    
    // group 0  =>  x[feature] <= cond
    // group 1  =>  x[feature] >  cond
    
    const int elements_group_1 = group_sizes[split_id + feature*NUM_SPLITS];
    const int elements_group_0 = rows - elements_group_1;
    
    
    
    // load number of counts of each class in group 1 / 0
    /* Scheme of memory:
     *            group 0       group 1
     * class 0    frac[1]       frac[0]
     * class 1    frac[3]       frac[2]
     * ...        ...           ...
     */
    real_t frac[2*NUM_CLASSES]; // rewrite to use 
    for (uint8_t it=0; it < NUM_CLASSES; it++)
    {
        frac[2*it]   =  (real_t) fractions[NUM_CLASSES*(split_id+feature*NUM_SPLITS) + it];
        frac[2*it+1] = ((real_t) storage[it]) - frac[2*it];
    }
    
#ifdef DEBUG2
    if (blockIdx.y == 0)
        printf("%4d %4d\tnumbers of 0:%4.0f/%4.0f\tgroup size: %4d/%4d\n",
               blockIdx.y,split_id,frac[1],frac[0],elements_group_0,elements_group_1);
    
    if (threadIdx.x == 0) printf("\n");
#endif
    
    // Evaluate fractions
    // TODO: Unroll loops
    // TODO: Get rid of conditions
    for (uint8_t it=0; it < NUM_CLASSES; it++)
    {
        frac[2*it]   = (elements_group_0 == 0) ? 0.0 : frac[2*it]   / ((real_t) elements_group_1);
        frac[2*it+1] = (elements_group_1 == 0) ? 0.0 : frac[2*it+1] / ((real_t) elements_group_0);
    }
    
#ifdef DEBUG2
    if (blockIdx.y == 0)
        printf("%4d %4d\tnumbers of 0:%1.4f/%1.4f\tgroup size: %4d/%4d\n",
               blockIdx.y,split_id,frac[1],frac[0],elements_group_0,elements_group_1);
    
    if (threadIdx.x == 0) printf("\n");
#endif
    
    // Find minimal information cost
    
    typedef cub::BlockReduce<real_t, BLOCK_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    
    __shared__ real_t best_inf;
    real_t max_inf;
    //real_t min_inf;
    real_t inf_cost;
    real_t best_cond = -999999999;
    
    if (split_id < NUM_SPLITS)
    {
        inf_cost = 0;
        for (uint8_t it=0; it < 2*NUM_CLASSES; it++)
        {
            inf_cost += cost_function(frac[it]);
        }
        
        //information += (1.0-zero_frac)*(1.0-zero_frac);
        //information = 1.0 - information;
    
    
#ifdef DEBUG
    if (blockIdx.y == 0)
        printf("%4d %4d\t%4.4f\n",blockIdx.y,split_id,inf_cost);
#endif
        
        // TODO: Very robust way, because cub::Min() does not work properlly... Fix it!!!
        inf_cost = 1 - inf_cost;
        
        //if (split_id < NUM_SPLITS)
        max_inf = BlockReduceT(temp_storage).Reduce(inf_cost, cub::Max());
        __syncthreads();
        
        
        //if (split_id < NUM_SPLITS)
        //min_inf = BlockReduceT(temp_storage).Reduce(inf_cost, cub::Min()); // no idea why not working ???
        //__syncthreads();
        
        if (threadIdx.x == 0) best_inf = max_inf;
        //    printf("feature: %d\tmin: %4.4f\tmax:%4.4f\n",blockIdx.y,min_inf,max_inf);
        __syncthreads();
        
        
        if (inf_cost == best_inf)
        {
            best_cond =  min_in_feature[feature];
            best_cond += (split_id+0.5)*(max_in_feature[feature] - best_cond)/((real_t) NUM_SPLITS);
#ifdef DEBUG
            printf("Best split in feature %d: %4d (x > %4.4f)\t(%4.4f)\n",feature,threadIdx.x,best_cond,1-inf_cost);
#endif
            
            // make use of existing arrays
            ((real_t*) storage)[2*feature]   = best_cond;
            ((real_t*) storage)[2*feature+1] = 1 -best_inf;
        }
    }
}





void DecisionTree::find_condition(unsigned* _feature, real_t* _cond, const unsigned group_rows, const unsigned data_offset)
{
    cuErrCheck(  cudaMemset(d_fractions, 0, sizeof(int)  * NUM_SPLITS*NUM_CLASSES*_cols)  );
    cuErrCheck(  cudaMemset(d_group_sizes, 0, sizeof(int)  * NUM_SPLITS*_cols)  );
    cuErrCheck(  cudaMemset(d_classes + _rows, 0, sizeof(uint8_t)  * (32 - _rows%32))  );       // TODO: Check if needed
    
    int* d_storage = (int*) d_workspace;
    
    
    int num_blocks = DIV_UP(group_rows,BLOCK_SIZE*ITEMS_PER_THREAD);
    if (group_rows%4)
    {
        kernel_find_classes_counts_tot<<< dim3(num_blocks,1,1), dim3(BLOCK_SIZE,1,1) >>>(d_classes, d_storage, _rows); // 9.6960us
    }
    else
    {
        kernel_find_classes_counts_tot1<<< dim3(num_blocks,1,1), dim3(BLOCK_SIZE,1,1) >>>((int*)d_classes, d_storage, group_rows); // 3.8080us, 2.5x faster
    }
    
    const dim3 blocks_grid( NUM_BLOCKS(group_rows), _cols, 1 );
    const dim3 block_size ( BLOCK_SIZE, 1, 1 );
    
    printf("blocks grid: (%4d,%4d,%4d)\n",blocks_grid.x,blocks_grid.y,blocks_grid.z);
    printf("block size:  (%4d,%4d,%4d)\n",block_size.x, block_size.y, block_size.z );
    
    kernel_find_fractions<<< blocks_grid, block_size >>>(d_data,d_classes, d_fractions, d_group_sizes,group_rows,_cols);
    cuErrCheck( cudaDeviceSynchronize() );
    
    
    int d_classes_counts[NUM_CLASSES];
    cuErrCheck(  cudaMemcpy( d_classes_counts, d_storage, sizeof(int) * NUM_CLASSES, cudaMemcpyDeviceToHost )  );
    
    
#ifdef DEBUG
    std::cout << std::endl;
    std::cout << "counts of  classes (device): " << std::endl;
    for (unsigned it=0 ; it < NUM_CLASSES; it++)
    {
        std::cout << d_classes_counts[it] << "  ";
    }
    std::cout << std::endl;
    
    std::cout << std::endl;
    std::cout << "counts of  classes (host): " << std::endl;
    for (unsigned it=0 ; it < NUM_CLASSES; it++)
    {
        std::cout << h_classes_counts[it] << "  ";
    }
    std::cout << std::endl;
    
    print_launch_params(_rows,_cols);
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    
    const unsigned fractions_per_row = NUM_SPLITS*NUM_CLASSES;
    int fractions[CHUNKS_PER_ROW*SPLITS_PER_BLOCK*NUM_CLASSES*_cols];
    
    cuErrCheck(  cudaMemcpy( fractions, d_fractions, sizeof(int) * fractions_per_row*_cols, cudaMemcpyDeviceToHost )  );
    
    std::cout << _cols << " / " << fractions_per_row << std::endl;
    
    unsigned feature;
    for (unsigned it=0; it < fractions_per_row; it++)
    {
        printf("%2u.\t",it);
        for ( feature=0; feature < _cols; feature++)
        {
            printf("  %4d\t", fractions[feature*fractions_per_row + it]);
        }
        printf("\n");
    }
#endif
    
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    
    kernel_find_split<<< dim3(1,_cols,1), dim3(NUM_SPLITS,1,1) >>>(d_fractions, d_storage, d_group_sizes, _rows);
    
    
    real_t conditions[2*_cols];
    cuErrCheck(  cudaMemcpy( conditions, (real_t*)d_storage, sizeof(real_t) * 2*_cols, cudaMemcpyDeviceToHost )  );
    
    real_t best_cond = NAN;
    real_t best_inf  = 1.0; // we assume that cost function has value in range [0.0,1.0]
    int best_feature =  -1;
    
    for (unsigned it=0; it <_cols; it++)
    {
        std::cout << "\tfeature: " << it << "\tcondition: " << conditions[2*it] << "\tinf.: " << conditions[2*it+1] << std::endl;
        if (conditions[2*it+1] < best_inf)
        {
            best_cond = conditions[2*it];
            best_inf  = conditions[2*it+1];
            best_feature = it;
        }
    }
    
    std::cout << std::endl;
    std::cout << "BEST:" << std::endl;
    std::cout << "\tfeature: " << best_feature << "\tcondition: " << best_cond << "\tinf.: " << best_inf << std::endl;
    
    
    *_cond    = best_cond;
    *_feature = (unsigned)best_feature;
}



// ================================================== PARTITIONIG ARRAY ==========================================================================


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





/*
 * 
 * We need to rearnage in linear memory...
 * | ----------------------------- |
 * | ----------------------------- |
 * | ----------------------------- |
 * | ----------------------------- |
 * 
 * Linear memory:                        Logical partition:
 * | xxxxx | xxxxxx| xxxxx | xxxxx |     | xxxxx |      | --------------------- |
 * | --------------------- | ------      | xxxxx |      | --------------------- |
 *  -------------- | --------------      | xxxxx |      | --------------------- |
 *  ------ | --------------------- |     | xxxxx |      | --------------------- |
 *
 * @param
 * 
 * @return            number of elements in group 0/1? 
 */
template <typename T1, typename T2>
inline  int make_partition( T1* d_classes,
                            T1* d_new_classes, 
                            T2* d_data, 
                            T2* d_new_data,
                            T2 condition,
                            const unsigned rows, const unsigned cols, const unsigned feature )
{
    int size_group_0, size_group_1;
    
    // first partiotion on classes and get number of elements in each group
    typename  thrust::detail::normal_iterator<thrust::device_ptr<T1>> last_copied;
    last_copied = thrust::copy_if(thrust::device_pointer_cast(d_classes),        // begining of data chunk to copy
                                  thrust::device_pointer_cast(d_classes+rows),   // end of data chunk to copy
                                  thrust::device_pointer_cast(d_data+feature*rows),   // the feature we want to check condition
                                  thrust::device_pointer_cast(d_new_classes),         // where to copy
                                  checkCondition<false,T2>(condition));
    size_group_0 = last_copied - 
                   typename thrust::device_vector<T1, thrust::device_malloc_allocator<T1>>::
                   iterator(thrust::device_pointer_cast(d_new_classes));
    size_group_1 = rows - size_group_0;
    
    thrust::remove_copy_if(       thrust::device_pointer_cast(d_classes),        // begining of data chunk to copy
                                  thrust::device_pointer_cast(d_classes+rows),   // end of data chunk to copy
                                  thrust::device_pointer_cast(d_data+feature*rows),   // the feature we want to check condition
                                  last_copied,                                        // where to copy
                                  checkCondition<false,T2>(condition));
    
    
    // TODO: use openmp and make default stream to be different for each host thread ?
    #pragma omp parallel for num_threads(4)
    for (unsigned it=0; it < cols; it++)
    {
        thrust::copy_if(        thrust::device_pointer_cast(d_data+it*rows),                     // begining of data chunk to copy
                                thrust::device_pointer_cast(d_data+(it+1)*rows),                 // end of data chunk to copy
                                thrust::device_pointer_cast(d_data+feature*rows),                // the feature we want to check condition
                                thrust::device_pointer_cast(d_new_data+it*size_group_0),         // where to copy
                                checkCondition<false,T2>(condition));
        thrust::remove_copy_if( thrust::device_pointer_cast(d_data+it*rows),                     // begining of data
                                thrust::device_pointer_cast(d_data+(it+1)*rows),                 // end of data
                                thrust::device_pointer_cast(d_data+feature*rows),                // feature
                                thrust::device_pointer_cast(d_new_data +
                                                            cols*size_group_0+it*size_group_1),  // iterator pointing to the end of 
                                checkCondition<false,T2>(condition));
    }
    
    
    
    return size_group_0;
}



// TODO:
/*
 * An attempt of solution that do not require additional sotrage...
 * Probably there is no way to make it work.
 * Even if we can make a global synchronization, what will happen when number of blocks is greater
 * that number of SMs and new blocks need to be loaded?
 *

__device__ partition_point[NUM_STREAMS]; // TODO: No more classes possible
__device__ sync_counter[NUM_STREAMS];

template <const int block_size,
          const int items_per_thread,
          const unsigned rows,
          const unsigned cols,
          const unsigned feature,
          const unsigned stream_id>
__global__ void kernel_partition(real_t*  d_data,
                                 int* d_classes,
                                 real_t cond)
{
    int tid = threadIdx
    int idx = block_size*blockIdx.x + tid;
    
    if (idx == 0 && blockIdx.y == 0) partition_point = 0; // clear memory
    if (idx == 1 && blockIdx.y == 0) sync_counter = 0; // clear memory
    
    uint8_t _classes[items_per_thread];
    real_t  _data[items_per_thread];
    //real_t _feature_data[items_per_thread];
    int conditions[items_per_thread];
    int prescanned[items_per_thread];
    int block_aggregate;
    
    __shared__ typename BlockScan::TempStorage temp_storage;
    
    
    // load features
    if (idx*items_per_thread < rows)
    {
        // Load a tile of data striped across threads
        cub::LoadDirectStriped<block_size>(tid, d_data+feature*rows, _data);
        
        
        // evaluate condition
        // #pragma unroll
        for (uin8_t it=0; it < items_per_thread; it++)
            conditions[it] = (int)(_data[it] > cond);
        __synchthreads();
        
        // make block-wide prescan on condtions with initial value -1 to evaluate indexes where to copy data
        BlockScan(temp_storage).ExclusiveSum(conditions, prescanned, block_aggregate);
        __synchthreads();
        
        // sum numbers of elements to be copied by each block
        if (tid == 0) atomicAdd(&partition_point,block_aggregate);
        __synchthreads();
        block_aggregate = partition_point; // to be known in each thread
        
        
        // load data to be copied, each thread will store a chunk of data
        /* NOTE: In this way we do not need additional array for input and output of partitioning,
                 but global synchronization required to be sure that all data loaded to kernels.*
        cub::LoadDirectStriped<block_size>(tid, d_data+blockIdx.y*rows, _data);
        if (blockIdx.y == 0)
        for (uint8_t it=0; it < items_per_thread/4; it++)
            ((int*) _classes)[it] = d_classes[idx*(items_per_thread/4) + it];
        __synchthreads();
        
        // global synchronization of all blocks needed
        if (tid == 0)
        {
            atomicAdd(&sync_counter, 1);
            while (sync_counter < gridDim.x*gridDim.y); // wait for every block to write to syncCounter???
        // possibly will cause deadlock...
        }
        __syncthreads();
        
        
        // make copy of group1
        // TODO: maybe radix sort here?
        for (uin8_t it=0; it < items_per_thread; it++)
        {
            if (conditions[it] == 1)
            {
                d_data[prescanned[it]+block_aggregate]    = _data[it];
            }
        }
        if (blockIdx.y == 0)
        for (uin8_t it=0; it < items_per_thread; it++)
        {
            if (conditions[it] == 1)
            {
                d_classes[prescanned[it]+block_aggregate] = _classes[it];
            }
        }
        //__syncthreads(); // TODO: Probably not necessary
        
        
        // negate conditions
        for (uin8_t it=0; it < items_per_thread; it++)
            conditions[it] = ~conditions[it];
        __synchthreads();
        
        // make block-wide prescan on condtions with initial value -1 to evaluate indexes where to copy data
        BlockScan(temp_storage).ExclusiveSum(conditions, prescanned);
        __synchthreads(); 
        
        
        // make copy of group1
        for (uin8_t it=0; it < items_per_thread; it++)
        {
            if (conditions[it] == 1)
            {
                d_data[prescanned[it]]    = _data[it];
            }
        }
        if (blockIdx.y == 0)
        for (uin8_t it=0; it < items_per_thread; it++)
        {
            if (conditions[it] == 1)
            {
                d_classes[prescanned[it]] = _classes[it];
            }
        }
    }
}*/




// ================================================== PARTITIONIG ARRAY ==========================================================================


void DecisionTree::build_tree_CART()
{
    
    unsigned *h_rows_array     = this->dtree_structure->h_rows;
    unsigned *h_features       = this->dtree_structure->h_features;
    unsigned *h_left_childreen = this->dtree_structure->h_left_childreen;
    real_t   *h_conditions     = this->dtree_structure->h_conditions;
    
    h_rows_array[0] = _rows; 
    
    
    unsigned node_id = 0;
    for (unsigned it=0; it < 4; it++)
    {
        // simpliest solution for partitioning
        cuErrCheck(  cudaMemcpy(d_classes_cpy, d_classes, sizeof(uint8_t) * _rows,       cudaMemcpyDeviceToDevice)  );
        cuErrCheck(  cudaMemcpy(d_data_cpy,    d_data,    sizeof(real_t)  * _rows*_cols, cudaMemcpyDeviceToDevice)  );
        
        // TODO: Here #pragma omp parallel for !!!
        for (unsigned jt =0; jt < pow(2,it); jt++)
        {
            if (h_rows_array[node_id] < ITEMS_PER_THREAD)
            {
                
                continue;
            }
            
            
            // musimy znac liczbe danych po podziale, by wiedziec o ile przesunac wskaznik!
            unsigned left_child = node_id + (pow(2,it) - jt) + (jt*pow(2,it)); 
            h_left_childreen[node_id] = left_child;
            
            unsigned data_offset = 0;
            for (unsigned kt=1; kt<=jt; kt++)
            {
                data_offset += h_rows_array[node_id - kt];
            }
            std::cout << "node: " << node_id << "\t rows: " << h_rows_array[node_id] << "\tdata_offset: " << data_offset << std::endl;
            
            this->find_condition( h_features   + node_id,
                                  h_conditions + node_id,
                                  h_rows_array[node_id],
                                  data_offset );
            
            real_t cond = h_conditions[node_id];
            int feature = h_features[node_id];
            
            // make partition
            int size_group_0;
            size_group_0 = make_partition<uint8_t,real_t>(d_classes_cpy,d_classes,d_data_cpy, d_data, cond,_rows,_cols,feature);
            
            
            h_rows_array[left_child]   = size_group_0;
            h_rows_array[left_child+1] = h_rows_array[node_id] - size_group_0;
            
            node_id++; // go to next node in binary tree
        }
    std::cout << "=====================================================================================================================" << std::endl;
    std::cout << std::endl;
        
        // next
        // TODO: rows = ...; // update number of rows!
        
    }
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    
    // print results
    std::cout << "rows at different levels:" << std::endl;
    node_id = 0;
    for (unsigned it=0; it < 2; it++)
    {
        for (unsigned jt =0; jt < pow(2,it); jt++)
        {
            std::cout << h_rows_array[node_id] << "\t";
            node_id++;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}




/*
 * More sophisticated solution of problem of building tree.
 * No additional copies needed but problem where the result data are after all...
 * 
void DecisionTree::build_tree_CART2()
{
    real_t*  data_handle[2]    = {d_data,d_data_cpy};
    uint8_t* classes_handle[2] = {d_classes,d_classes_cpy};
    unsigned node_id = 0;
    for (unsigned it=0; it < 1; it++)
    {
        // TODO: Here #pragma omp parallel for !!!
        for (unsigned jt =0; jt < pow(2,it); jt++)
        {
            // musimy znac liczbe danych, by wiedziec o ile przesunac wskaznik!
            
            
            
            real_t cond;
            int feature;
            this->find_condition(&feature,&cond);
            
            
            // make partition
            int size_group_0;
            size_group_0 = make_partition_one(classes_handle[it%2],classes_handle[(it+1)%2],d_data,cond,_rows,_cols,feature);
            size_group_0 = make_partition_many(data_handle[it%2],data_handle[(it+1)%2],cond,_rows,_cols,feature);
            
            // reorganize classes array
            
            node_id++;
        }
        
        // next
        // TODO: rows = ...; // update number of rows!
        if (_rows < 2) break;
    }
}
 */






void DecisionTree::one_split()
{
    
    unsigned *h_rows_array     = this->dtree_structure->h_rows;
    unsigned *h_features       = this->dtree_structure->h_features;
    unsigned *h_left_childreen = this->dtree_structure->h_left_childreen;
    real_t   *h_conditions     = this->dtree_structure->h_conditions;
    
    h_rows_array[0] = _rows; 
    
    
    unsigned node_id = 0;
    for (unsigned it=0; it < 1; it++)
    {
        // simpliest solution for partitioning
        cuErrCheck(  cudaMemcpy(d_classes_cpy, d_classes, sizeof(uint8_t) * _rows,       cudaMemcpyDeviceToDevice)  );
        cuErrCheck(  cudaMemcpy(d_data_cpy,    d_data,    sizeof(real_t)  * _rows*_cols, cudaMemcpyDeviceToDevice)  );
        
        // TODO: Here #pragma omp parallel for !!!
        if (h_rows_array[node_id] < ITEMS_PER_THREAD)
        {
            continue;
        }
        
        // musimy znac liczbe danych po podziale, by wiedziec o ile przesunac wskaznik!
        unsigned left_child = 1; 
        h_left_childreen[node_id] = left_child;
        
        this->find_condition( h_features+node_id,
                              h_conditions+node_id,
                              h_rows_array[node_id] );
        
        real_t cond = h_conditions[node_id];
        int feature = h_features[node_id];
        
        // make partition
        int size_group_0;
        size_group_0 = make_partition<uint8_t,real_t>(d_classes_cpy,d_classes,d_data_cpy, d_data, cond,_rows,_cols,feature);
        
        h_rows_array[left_child]   = size_group_0;
        h_rows_array[left_child+1] = h_rows_array[node_id] - size_group_0;
        
        node_id++; // go to next node in binary tree
    }
    std::cout << "=====================================================================================================================" << std::endl;
    std::cout << std::endl;
    
    // next
    // TODO: rows = ...; // update number of rows!
        
    
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;
    
    // print results
    std::cout << "rows at different levels:" << std::endl;
    node_id = 0;
    for (unsigned it=0; it < 2; it++)
    {
        for (unsigned jt =0; jt < pow(2,it); jt++)
        {
            std::cout << h_rows_array[node_id] << "\t";
            node_id++;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}










#endif