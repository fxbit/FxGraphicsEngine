//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

typedef unsigned int uint;

extern "C" __global__ 
void memfill( uint *Dest,
              uint  Dest_len,
              uint  Offset,
              uint *Fill_pattern,
              uint  pattern_len,
              uint  Fill_len)
{
    const uint X = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( X >= (Dest_len - Offset) || X >= Fill_len)
        return;
    
    Dest[X + Offset] = Fill_pattern[X % pattern_len];
}
             
