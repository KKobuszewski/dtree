#ifndef __DECISION_TREE_HPP__
#define __DECISION_TREE_HPP__


#include <common_settings.h>
#include <stdlib.h>
#include <stdio.h>

#include <cuerror.cuh>


// ========================== device variables =================

__constant__ real_t max_in_feature[MAX_FEATURES];
__constant__ real_t min_in_feature[MAX_FEATURES];



#ifdef DEBUG
#include <debug_utils.cuh>
#endif



//__device__ real_t workspace[WORKSPACE_ELEMENTS]; // some additional memory to share data between blocks



//template <const size_t N>
class DecisionTree
{
private:
    static unsigned class_idx;
    
    // handles for data and classes
    real_t*  h_data;
    uint8_t* h_classes;
    real_t*  d_data;
    real_t*  d_data_cpy;
    uint8_t* d_classes;
    uint8_t* d_classes_cpy;
    real_t*  h_max_in_feature;
    real_t*  h_min_in_feature;
    int*  h_fractions;
    int*  d_fractions;
    int*  d_group_sizes;
    
    real_t*  h_classes_counts;
    
    
    // some additional workspace ???
    void* d_workspace;
    const size_t workspace_size = 1<<20; // 2**20 bytes
    
    // files with data
    FILE *datafile;
    FILE *classesfile;
    
    // cuda streams
    cudaStream_t streams[NUM_STREAMS];
    
    
    
    // dtree structure
    struct DTreeStructure
    {
        unsigned *h_rows;
        unsigned *h_features;
        unsigned *h_left_childreen;
        real_t   *h_conditions;
        
        DTreeStructure(const unsigned max_depth)
        {
            h_rows           = new unsigned[(int)pow(max_depth+1,2)-1];
            h_features       = new unsigned[(int)pow(max_depth+1,2)-1];
            h_left_childreen = new unsigned[(int)pow(max_depth+1,2)-1];
            h_conditions     = new real_t  [(int)pow(max_depth+1,2)-1];
        }
        
        ~DTreeStructure()
        {
            delete h_rows;
            delete h_features;
            delete h_left_childreen;
            delete h_conditions;
        }
    };
    
public:
    // dtree info
    const unsigned _max_depth;
    unsigned* h_left_child;
    unsigned* d_left_child;
    real_t*   h_split_conditions;
    real_t*   d_split_conditions;
    
    
    DTreeStructure* dtree_structure;
    
    // data info
    const unsigned _rows; //  = 1372
    const unsigned _cols; //  = 4
    
    // constructors
    DecisionTree(unsigned rows, unsigned cols, unsigned max_depth  = 10):
                _rows(rows),_cols(cols),_max_depth(max_depth)
    {
        if (_cols > MAX_FEATURES)
        {
            fprintf(stderr,"ERROR! Feature index greater than number of data cols! (file: %s, line: %d)\n",__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }
        if (NUM_CLASSES < 2 || NUM_CLASSES > MAX_CLASSES)
        {
            fprintf(stderr,"Error! Wrong number of classes: %d (file: %s, line: %d)\n",NUM_CLASSES,__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }
        class_idx++;
        if (class_idx > MAX_DTREES)
        {
            fprintf(stderr,"Error! Too many instances of class DecisoinTree: %d/%d (file: %s, line: %d)\n",class_idx, MAX_DTREES,__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }
        
        // allocate memory
        
        size_t rows_aligned = _rows + (32 - _rows%32);
        
        cuErrCheck(  cudaHostAlloc(&h_data,      sizeof(real_t)  * _rows*_cols, cudaHostAllocWriteCombined)  );
        cuErrCheck(  cudaHostAlloc(&h_classes,   sizeof(uint8_t) * _rows, cudaHostAllocWriteCombined)        );
        cuErrCheck(  cudaHostAlloc(&h_fractions, sizeof(int)     * NUM_SPLITS*NUM_CLASSES*_cols, 
                                                                                      cudaHostAllocDefault)  );
        
        cuErrCheck(  cudaMalloc(&d_data,        sizeof(real_t)  * _rows*_cols)                   );
        cuErrCheck(  cudaMalloc(&d_data_cpy,    sizeof(real_t)  * _rows*_cols)                   );
        cuErrCheck(  cudaMalloc(&d_classes,     sizeof(uint8_t) * rows_aligned)                  );
        cuErrCheck(  cudaMalloc(&d_classes_cpy, sizeof(uint8_t) * rows_aligned)                  );
        cuErrCheck(  cudaMalloc(&d_fractions,   sizeof(int)     * NUM_SPLITS*NUM_CLASSES*_cols)  );
        cuErrCheck(  cudaMalloc(&d_group_sizes, sizeof(int)     * NUM_SPLITS*_cols)  );
        cuErrCheck(  cudaMalloc(&d_workspace, workspace_size)  );
        
        
        cuErrCheck(  cudaMemset(d_classes, 0, sizeof(uint8_t) * rows_aligned) );
        std::cout << "rows aligned: " << rows_aligned << std::endl;
        
        h_max_in_feature = new real_t[_cols];
        h_min_in_feature = new real_t[_cols];
        
        h_classes_counts = new real_t[NUM_CLASSES];
        for (unsigned it=0; it < NUM_CLASSES; it++)
            h_classes_counts[it] = 0;
        
        
        // initialize many streams
        #pragma omp parallel for
        for (unsigned it=0; it < NUM_STREAMS; it++)
        {
            cuErrCheck(  cudaStreamCreate(&streams[it])  );
        }
        
        
        dtree_structure = new DTreeStructure(MAX_DEPTH);
    }
    
    // destructors
    ~DecisionTree()
    {
        // clean up memory
        delete h_max_in_feature;
        delete h_min_in_feature;
        delete h_classes_counts;
        
        cuErrCheck(  cudaFreeHost(h_data)       );
        cuErrCheck(  cudaFreeHost(h_classes)    );
        cuErrCheck(  cudaFreeHost(h_fractions)  );
        
        cuErrCheck(  cudaFree(d_data)       );
        cuErrCheck(  cudaFree(d_data_cpy)   );
        cuErrCheck(  cudaFree(d_classes)    );
        cuErrCheck(  cudaFree(d_fractions)  );
        
        // finalize streams
        #pragma omp parallel for
        for (unsigned it=0; it < NUM_STREAMS; it++)
        {
            cuErrCheck(  cudaStreamDestroy(streams[it])  );
        }
        
        delete dtree_structure;
    }
    
    
    
    
    // public methods
    void load_data(const char* datafilename, const char* classesfilename)
    {
        // read data from files
        int bytes_read = 0;
        datafile    = fopen(datafilename,"rb");
        if (!datafile)     { fprintf(stderr,"Error opening data file: %s!\n",datafilename);     exit(EXIT_FAILURE); }
        bytes_read = fread(h_data,sizeof(real_t),_rows*_cols,datafile);
        std::cout << "bytes read: " << bytes_read << std::endl;
        fclose(datafile);
        
        classesfile = fopen(classesfilename,"rb");
        if (!classesfile)  { fprintf(stderr,"Error opening data file: %s!\n",classesfilename);  exit(EXIT_FAILURE); }
        bytes_read = fread(h_classes,sizeof(real_t),_rows,classesfile);
        std::cout << "bytes read: " << bytes_read << std::endl;
        fclose(classesfile);
        
        
        // copy data to device
        cuErrCheck(  cudaMemcpyAsync(d_data,    h_data,    sizeof(real_t)  * _rows*_cols, cudaMemcpyHostToDevice, streams[0])  );
        cuErrCheck(  cudaMemcpyAsync(d_classes, h_classes, sizeof(uint8_t) * _rows      , cudaMemcpyHostToDevice, streams[1])  );
        
        
        for (unsigned it=0; it < _rows; it++)
        {
            real_t val = h_classes[it];
            for (unsigned jt=0; jt < NUM_CLASSES; jt++)
            {
                if (val == jt) h_classes_counts[jt] += 1;
            }
        }
        
    }
    
    // TODO: make memmap
    
    
    void find_max_min_per_feature()
    {
#ifdef DEBUG 
        printf("\n\n\n");
        printf("# MIN/MAX ELEMENTS (HOST):\n");
#endif
        // TODO: Enable many classes -> more constant memory and changing pointer by class_idx!
        // TODO: Copying device -> device (no need to copy )
        // TODO: #pragma omp parallel for num_treads(2), divide on streams!!!
        for (unsigned it=0; it < _cols; it++)
        {
            if (it >= MAX_FEATURES) { fprintf(stderr,"Error! Too many features in data point!\n"); exit(EXIT_FAILURE); }
            
            std::cout << std::fixed << it  << ". ";
            
            // find max
            h_max_in_feature[it] = *(thrust::max_element(thrust::device_pointer_cast<real_t>(d_data+it*_rows),
                                                         thrust::device_pointer_cast<real_t>(d_data+(it+1)*_rows) ));
            cuErrCheck( cudaMemcpyToSymbol(max_in_feature,&h_max_in_feature[it], sizeof(real_t),it*sizeof(real_t)) );
#ifdef DEBUG
            std::cout << "max: " << h_max_in_feature[it] << "\t";
#endif
            
            
            // find min
            h_min_in_feature[it] = *(thrust::min_element(thrust::device_pointer_cast<real_t>(d_data+it*_rows),
                                                         thrust::device_pointer_cast<real_t>(d_data+(it+1)*_rows) ));
            cuErrCheck( cudaMemcpyToSymbol(min_in_feature, &h_min_in_feature[it], sizeof(real_t),it*sizeof(real_t)) );
#ifdef DEBUG
            std::cout << "min: " << h_min_in_feature[it] <<std::endl;
#endif
        }
        
#ifdef DEBUG
        printf("\n");
        printf("# MIN/MAX ELEMENTS (DEVICE):\n");
        kernel_print_const_mem<<<1,1>>>(_cols);
        cudaDeviceSynchronize();
        printf("\n\n\n");
#endif
    }
    
    
    
    void find_condition(unsigned* _feature, real_t* _cond);
    
    void build_tree_CART();
    void build_tree_CART2();
    
    
    
    
    
    // getters
    real_t* get_data()
    {
        return h_data;
    }
    
    
    uint8_t* get_classes()
    {
        return h_classes;
    }
    
    
    void import_data_to_host()
    {
        cuErrCheck(  cudaMemcpy( h_data,    d_data,    sizeof(real_t)  * _rows*_cols, cudaMemcpyDeviceToHost )  );
        cuErrCheck(  cudaMemcpy( h_classes, d_classes, sizeof(uint8_t) * _rows,       cudaMemcpyDeviceToHost )  );
    }
    
};   // end class DecisionTree



unsigned DecisionTree::class_idx = 0;



/*
How to implement more types for this class...

template <typename T1, typename T2, typename T3>
class DecisionTree { ... };

template <typename T1, typename T2, typename T3>
void allocate_cuda_memory(T1** d_inputs_data_type1, const unsigned inputs_num_type1,
                          T2** d_inputs_data_type1, const unsigned inputs_num_type2,
                          T3** d_inputs_data_type1, const unsigned inputs_num_type3)
{
    
    
}
*/


#endif