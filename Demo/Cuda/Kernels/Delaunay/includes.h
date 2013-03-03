
#ifndef H_INCLUDES
#define H_INCLUDES

/* global defines */
#define UNSET 0xFFFFFFFF

/* gloabal parameters */
#define MAX_FACE_CORRECTIONS 50

/* set the using of shared mem for stack or not */
//#define USE_SHARED_MEM
#define STACK_MAX_NUM 64
#define MERGE_VERTICAL_X 4
#define MERGE_VERTICAL_Y 8

//#define USE_BOUNDARY

/* include Code */
#include "structs.h"

/* TriangleUtils.h */
__device__ bool SideTest(float2 orig , float2 A, float2 B);

#endif
