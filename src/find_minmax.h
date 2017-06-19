#ifndef __FIND_MINMAX_H__
#define __FIND_MINMAX_H__










void find_max_min_inside_features(real_t* d_data_mem, const unsigned rows, const unsigned cols)
{
    //real_t 
    real_t x;
    
    for (unsigned it=0; it < cols; it++)
    {
        if (it >= MAX_FEATURES) { fprintf(stderr,"Error! Too many features in data point!\n"); exit(EXIT_FAILURE); }
        
        std::cout << std::fixed << it  << ". ";
        
        // find max
        x = *(thrust::max_element(thrust::device_pointer_cast<real_t>(d_data_mem+it*rows),
                                         thrust::device_pointer_cast<real_t>(d_data_mem+(it+1)*rows) ));
        cuErrCheck( cudaMemcpyToSymbol(max_in_feature,&x, sizeof(real_t),it*sizeof(real_t)) );
        std::cout << "max: " << x << "\t";
        
        
        // find min
        x = *(thrust::min_element(thrust::device_pointer_cast<real_t>(d_data_mem+it*rows),
                                         thrust::device_pointer_cast<real_t>(d_data_mem+(it+1)*rows) ));
        cuErrCheck( cudaMemcpyToSymbol(min_in_feature, &x, sizeof(real_t),it*sizeof(real_t)) );
        std::cout << "min: " << x <<std::endl;
    }
    
    
    printf("\n\n\n");
    printf("# MIN/MAX ELEMENTS ALONG DISTINCT FEATURES:\n");
    kernel_print_const_mem<<<1,1>>>(cols);
    printf("\n\n\n");
}










#endif