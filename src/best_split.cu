

#define DEBUG

// =================== custom includes =====================================

#include <common_settings.h>
#include <cuerror.cuh>           // checking error codes
#include <DecisionTree.hpp>
#include <split_condition.h>     // 

// ===================







/*
 * 
 */
int main(int argc, char* argv[])
{
    // dtree info
    const unsigned max_depth = 10;
    
    
    // data info
    const int rows = 1372;
    const int cols = 4;
    
    // create instantion of DTree
    DecisionTree* dtree = new DecisionTree(rows,cols);
    
    // read data
#ifndef USE_DOUBLE
    dtree->load_data("data_banknote/data_banknote_authentication_flt.bin","data_banknote/classes.bin");
#else
    dtree->load_data("data_banknote/data_banknote_authentication_dbl.bin","data_banknote/classes.bin");
#endif
    
    real_t*  h_data    = dtree->get_data();
    uint8_t* h_classes = dtree->get_classes();
    
    std::cout << std::fixed;
    unsigned jj;
    for (unsigned ii=0; ii < rows; ii+=4)
    {
        std::cout << ii << ".\t";
        for (jj=0; jj < cols; jj++)
            std::cout << h_data[ii+jj*rows] << "\t";
        std::cout << ((unsigned) h_classes[ii]) << std::endl;
    }
    std::cout << "max\t";
    for (jj=0; jj < cols; jj++)
        std::cout << *std::max_element(h_data+jj*rows,
                                       h_data+(jj+1)*rows) << "\t";
    std::cout << std::endl;
    std::cout << "min\t";
    for (jj=0; jj < cols; jj++)
        std::cout << *std::min_element(h_data+jj*rows,
                                       h_data+(jj+1)*rows) << "\t";
    std::cout << std::endl;
    
    // in this case last column of data is classes number
    
    
    
    
    // find min/max
    dtree->find_max_min_per_feature();
    
    std::cout << std::endl;
    
    for (unsigned it=0; it < 1; it++)
        dtree->find_condition();
    
    /*
    unsigned feature = 0;
    find_condition(thrust::raw_pointer_cast(&d_data_mem[0]),
                   thrust::raw_pointer_cast(&d_classes[0]),
                   rows, cols, feature);
    
    */
    
    delete dtree;
    
    return EXIT_SUCCESS;
}