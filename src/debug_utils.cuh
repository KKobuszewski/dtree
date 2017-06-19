#ifndef __DEBUG_UTILS_H__
#define __DEBUG_UTILS_H__


__global__ void kernel_print_const_mem(const unsigned cols)
{
    for (unsigned it=0; it < cols; it++)
    {
        printf("max: %f\tmin: %f\n",max_in_feature[it],min_in_feature[it]);
    }
}



#endif