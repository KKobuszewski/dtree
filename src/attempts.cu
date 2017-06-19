/*
 * @param real_t* data   - pointer to data array
 * @param real_t 
 * 
 * 
 * total amount of smem = block_size*NUM_SPLITS + ...
 * So max block_size = 
 * 
 * To be used  <<< (NUM_BLOCKS(rows),cols,1), (BLOCK_SIZE,1,1) >>>
 * NOTE: Second dimension of grid represents features!
 * 
 */
__global__ void kernel_find_fractions_bugged(real_t* data, uint8_t* classes,              /*dataset*/
                                            real_t* fractions,                           /*storage*/
                                            const unsigned rows, const unsigned cols)    /*dimensions*/
{
    //int  tid = threadIdx.x*ITEMS_PER_THREAD;        // thread id inside block
    const int data_id = threadIdx.x*ITEMS_PER_THREAD + 
                        ITEMS_PER_BLOCK*( blockIdx.x%BLOCKS_PER_CHUNK(rows) );                          // id of data element to be read, TODO: no modulo!!!
    
    const unsigned feature = blockIdx.y;                                                                     // second dimension of grid iterates over features! 
    unsigned       frac_id = feature*NUM_SPLITS*(NUM_CLASSES-1) +                                            // move index in dependence on which feature we process
                             blockIdx.x/BLOCKS_PER_CHUNK(rows);                     // add split index in row
    
    
    
    // allocate memory
    typedef cub::BlockReduce<uint16_t, BLOCK_SIZE, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceT;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    
    uint16_t _conditions[ITEMS_PER_THREAD] = {0}; // NOTE: uint8_t can give overflow, sadly...
    uint8_t  _classes[ITEMS_PER_THREAD] = {0};
    real_t   _data[ITEMS_PER_THREAD] = {0};
    
    real_t cond, df;
    cond =  min_in_feature[feature];                                                                         // each block operates on one feature
    df   = (max_in_feature[feature] - cond)/((real_t) NUM_SPLITS);                                           // so this is the same for every split in chunk
    
    //if ( data_id < rows )
    if ( data_id + ITEMS_PER_THREAD <= rows )
    {
    
    // load <- done once, because every chunk and split need to be evaluated on same data
    for (uint8_t it=0; it < ITEMS_PER_THREAD; it++)
        _classes[it] = classes[ data_id + it ];
    
    for (uint8_t it=0; it < ITEMS_PER_THREAD; it++)
        _data[it]    = data[ feature*rows + data_id + it ];
    
    
    // TODO: Unroll this loops
    for (unsigned it_split = 0;   it_split   < SPLITS_PER_BLOCK; it_split++)                                 // enables evaluation of few splits inside same block
    for (unsigned it_classes = 1; it_classes < NUM_CLASSES;      it_classes++)                               // split must be evaluated for each class
    {
        /*
        frac_id = feature*NUM_SPLITS*(NUM_CLASSES-1) +                        
                   CHNUK_ID(block_id_x,rows)*SPLITS_PER_BLOCK*(NUM_CLASSES-1) +
                   it_split*(NUM_CLASSES-1) + (it_classes-1);
        */
        
        // find split condition: this is 
        cond += df*( CHNUK_ID(blockIdx.x,rows)*SPLITS_PER_BLOCK + it_split  );
        
        // evaluate conditions
#if (NUM_CLASSES == 2)
        for (uint8_t it=0; it < ITEMS_PER_THREAD; it++)
            _conditions[it] = (uint16_t)(_classes[it] && (_data[it] > cond)); // check only if class is 1
#elif (NUM_CLASSES > 2)
        for (uint8_t it=0; it < ITEMS_PER_THREAD; it++)
            _conditions[it] = (_classes[it] == it_classes) && (_data[it] > cond); // enables many classes
#endif
        
        __syncthreads(); // check if needed?
        
        // Make reduction on conditions and find fractions
        // NOTE: there are BLOCKS_PER_CHUNK writing to same place in global mem
        // NOTE: fractions must be set to zeros before!!!
        real_t sum = 0;
        //if (CHNUK_ID(blockIdx.x,rows) < 1 && threadIdx.x == 0)
        //    printf("%4d %4d (%d)   (x >%f)  sum: %f\n", threadIdx.x, blockIdx.x, feature, cond, sum);
        
        sum = ((real_t) BlockReduceT(temp_storage).Sum(_conditions));// / ((real_t) rows);
        
        __syncthreads();
        
        if (blockIdx.x < 32 && feature==0 && threadIdx.x == 0 && it_split == 0)
            printf("block: %4d  feature:%4d  frac_id:%4d   (x >%.2f) \t sum: %.0f\n", blockIdx.x, feature, frac_id, cond, sum);
        
        if (threadIdx.x == 0)
            atomicAdd(&fractions[frac_id], sum);
        // TODO: check if two blocks read/write from __global__ not simultaneously !!! <- in another case this can generate undefined behaviour
        
#ifdef DEBUG2
        if (threadIdx.x == 0 && blockIdx.x % BLOCKS_PER_CHUNK(rows))
        {
            
            printf("%4d. (%d)\tsplit:%2d\tcondition: x > %f\tfraction: %f\n",blockIdx.x,feature,frac_id,cond,fractions[frac_id]);
        }
#endif
        
        
        
        frac_id++;
    }
    
#if (SPLITS_PER_BLOCK > 1)
    // fractions must be added inside each split?   <- NO!
    
    
#else
    
#endif
    
    
    }
}