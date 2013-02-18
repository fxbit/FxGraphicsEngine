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

typedef struct{
    float x;
    float y;
} FxVector2f;

#define SHARED_SIZE_LIMIT 1024U
#define     SAMPLE_STRIDE 128

#define KEY_TYPE float2

template<typename T> __device__ int compare_l(T A, T B);
template<typename T> __device__ int compare_le(T A, T B);
template<typename T> __device__ int compare_g(T A, T B);
template<typename T> __device__ int compare_ge(T A, T B);

template<> __device__ int compare_l(KEY_TYPE A, KEY_TYPE B) { return A.x<B.x;}
template<> __device__ int compare_g(KEY_TYPE A, KEY_TYPE B) { return A.x>B.x;}
template<> __device__ int compare_le(KEY_TYPE A, KEY_TYPE B) { return A.x<=B.x;}
template<> __device__ int compare_ge(KEY_TYPE A, KEY_TYPE B) { return A.x>=B.x;}

template<> __device__ int compare_l(uint A, uint B) { return A<B;}
template<> __device__ int compare_g(uint A, uint B) { return A>B;}
template<> __device__ int compare_le(uint A, uint B) { return A<=B;}
template<> __device__ int compare_ge(uint A, uint B) { return A>=B;}


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
static inline __host__ __device__ uint iDivUp(uint a, uint b)
{
    return ((a % b) == 0) ? (a / b) : (a / b + 1);
}

static inline __host__ __device__ uint getSampleCount(uint dividend)
{
    return iDivUp(dividend, SAMPLE_STRIDE);
}

#define W (sizeof(uint) * 8)
static inline __device__ uint nextPowerOfTwo(uint x)
{
    /*
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    */
    return 1U << (W - __clz(x - 1));
}


template<uint sortDir,typename T> static inline __device__ 
uint binarySearchExclusive(T val, T *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }
    
    uint pos = 0;
    
    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);
    
        if ((sortDir && compare_l<T>(data[newPos - 1],val)) || (!sortDir && compare_g<T>(data[newPos - 1],val)))
        {
            pos = newPos;
        }
    }
    
    return pos;
}
    
template<uint sortDir,typename T> static inline __device__ 
uint binarySearchInclusive(T val, T *data, uint L, uint stride)
{
    if (L == 0)
    {
        return 0;
    }
        
    uint pos = 0;
        
    for (; stride > 0; stride >>= 1)
    {
        uint newPos = umin(pos + stride, L);
            
        if ((sortDir && compare_le<T>(data[newPos - 1],val)) || (!sortDir && compare_ge<T>(data[newPos - 1],val)))
        {
            pos = newPos;
        }
    }
        
    return pos;
}

////////////////////////////////////////////////////////////////////////////////

extern "C"  
__global__ void generateSampleRanksKernelUp(uint *d_RanksA,
                                            uint *d_RanksB,
                                            KEY_TYPE *d_SrcKey,
                                            uint stride,
                                            uint N,
                                            uint threadCount)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_SrcKey += segmentBase;
    d_RanksA += segmentBase / SAMPLE_STRIDE;
    d_RanksB += segmentBase / SAMPLE_STRIDE;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint segmentSamplesA = getSampleCount(segmentElementsA);
    const uint segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        d_RanksA[i] = i * SAMPLE_STRIDE;
        d_RanksB[i] = binarySearchExclusive<1, KEY_TYPE>(
                          d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,
                          segmentElementsB, nextPowerOfTwo(segmentElementsB)
                      );
    }

    if (i < segmentSamplesB)
    {
        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<1, KEY_TYPE>(
                    d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,
                    segmentElementsA, nextPowerOfTwo(segmentElementsA)
                );
    }
}
    

extern "C"  
__global__ void generateSampleRanksKernelDown(uint *d_RanksA,
        uint *d_RanksB,
        KEY_TYPE *d_SrcKey,
        uint stride,
        uint N,
        uint threadCount)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_SrcKey += segmentBase;
    d_RanksA += segmentBase / SAMPLE_STRIDE;
    d_RanksB += segmentBase / SAMPLE_STRIDE;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint segmentSamplesA = getSampleCount(segmentElementsA);
    const uint segmentSamplesB = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        d_RanksA[i] = i * SAMPLE_STRIDE;
        d_RanksB[i] = binarySearchExclusive<0, KEY_TYPE>(
                          d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride,
                          segmentElementsB, nextPowerOfTwo(segmentElementsB)
                      );
    }

    if (i < segmentSamplesB)
    {
        d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
        d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<0, KEY_TYPE>(
                    d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0,
                    segmentElementsA, nextPowerOfTwo(segmentElementsA)
                );
    }
}

extern "C"  
__global__ void mergeSortSharedKernelUp(KEY_TYPE *d_DstKey,
										uint *d_DstVal,
										KEY_TYPE *d_SrcKey,
										uint *d_SrcVal,
										uint arrayLength)
{
	__shared__ KEY_TYPE s_key[SHARED_SIZE_LIMIT];
	__shared__ uint s_val[SHARED_SIZE_LIMIT];

	d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
	s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
	s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
	s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

	for (uint stride = 1; stride < arrayLength; stride <<= 1)
	{
		uint     lPos = threadIdx.x & (stride - 1);
		KEY_TYPE *baseKey = s_key + 2 * (threadIdx.x - lPos);
		uint *baseVal = s_val + 2 * (threadIdx.x - lPos);

		__syncthreads();
		KEY_TYPE keyA = baseKey[lPos +      0];
		uint valA = baseVal[lPos +      0];
		KEY_TYPE keyB = baseKey[lPos + stride];
		uint valB = baseVal[lPos + stride];
		uint posA = binarySearchExclusive<1, KEY_TYPE>(keyA, baseKey + stride, stride, stride) + lPos;
		uint posB = binarySearchInclusive<1, KEY_TYPE>(keyB, baseKey +      0, stride, stride) + lPos;

		__syncthreads();
		baseKey[posA] = keyA;
		baseVal[posA] = valA;
		baseKey[posB] = keyB;
		baseVal[posB] = valB;
	}
		 
	__syncthreads();
	d_DstKey[                      0] = s_key[threadIdx.x +                       0];
	d_DstVal[                      0] = s_val[threadIdx.x +                       0];
	d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
	d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

extern "C"  
__global__ void mergeSortSharedKernelDown(KEY_TYPE *d_DstKey,
                                          uint *d_DstVal,
                                          KEY_TYPE *d_SrcKey,
                                          uint *d_SrcVal,
                                          uint arrayLength)
{
	__shared__ KEY_TYPE s_key[SHARED_SIZE_LIMIT];
	__shared__ uint s_val[SHARED_SIZE_LIMIT];

	d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
	s_key[threadIdx.x +                       0] = d_SrcKey[                      0];
	s_val[threadIdx.x +                       0] = d_SrcVal[                      0];
	s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
	s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

	for (uint stride = 1; stride < arrayLength; stride <<= 1)
	{
		uint     lPos = threadIdx.x & (stride - 1);
		KEY_TYPE *baseKey = s_key + 2 * (threadIdx.x - lPos);
		uint *baseVal = s_val + 2 * (threadIdx.x - lPos);

		__syncthreads();
		KEY_TYPE keyA = baseKey[lPos +      0];
		uint valA = baseVal[lPos +      0];
		KEY_TYPE keyB = baseKey[lPos + stride];
		uint valB = baseVal[lPos + stride];
		uint posA = binarySearchExclusive<0, KEY_TYPE>(keyA, baseKey + stride, stride, stride) + lPos;
		uint posB = binarySearchInclusive<0, KEY_TYPE>(keyB, baseKey +      0, stride, stride) + lPos;

		__syncthreads();
		baseKey[posA] = keyA;
		baseVal[posA] = valA;
		baseKey[posB] = keyB;
		baseVal[posB] = valB;
	}
    
	__syncthreads();
	d_DstKey[                      0] = s_key[threadIdx.x +                       0];
	d_DstVal[                      0] = s_val[threadIdx.x +                       0];
	d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
	d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////

extern "C"
__global__ void mergeRanksAndIndicesKernel( uint *d_Limits,
                                            uint *d_Ranks,
                                            uint stride,
                                            uint N,
                                            uint threadCount)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= threadCount)
    {
        return;
    }

    const uint           i = pos & ((stride / SAMPLE_STRIDE) - 1);
    const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
    d_Ranks  += (pos - i) * 2;
    d_Limits += (pos - i) * 2;

    const uint segmentElementsA = stride;
    const uint segmentElementsB = umin(stride, N - segmentBase - stride);
    const uint segmentSamplesA  = getSampleCount(segmentElementsA);
    const uint segmentSamplesB  = getSampleCount(segmentElementsB);

    if (i < segmentSamplesA)
    {
        uint dstPos = binarySearchExclusive<1U, uint>(d_Ranks[i], d_Ranks + segmentSamplesA, segmentSamplesB, nextPowerOfTwo(segmentSamplesB)) + i;
        d_Limits[dstPos] = d_Ranks[i];
    }

    if (i < segmentSamplesB)
    {
        uint dstPos = binarySearchInclusive<1U, uint>(d_Ranks[segmentSamplesA + i], d_Ranks, segmentSamplesA, nextPowerOfTwo(segmentSamplesA)) + i;
        d_Limits[dstPos] = d_Ranks[segmentSamplesA + i];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 3: merge elementary intervals
////////////////////////////////////////////////////////////////////////////////
template<uint sortDir> inline __device__ void merge(
    KEY_TYPE *dstKey,
    uint *dstVal,
    KEY_TYPE *srcAKey,
    uint *srcAVal,
    KEY_TYPE *srcBKey,
    uint *srcBVal,
    uint lenA,
    uint nPowTwoLenA,
    uint lenB,
    uint nPowTwoLenB
)
{
	KEY_TYPE keyA, keyB;
    uint valA, valB, dstPosA, dstPosB;

    if (threadIdx.x < lenA)
    {
        keyA = srcAKey[threadIdx.x];
        valA = srcAVal[threadIdx.x];
        dstPosA = binarySearchExclusive<sortDir>(keyA, srcBKey, lenB, nPowTwoLenB) + threadIdx.x;
    }

    if (threadIdx.x < lenB)
    {
        keyB = srcBKey[threadIdx.x];
        valB = srcBVal[threadIdx.x];
        dstPosB = binarySearchInclusive<sortDir>(keyB, srcAKey, lenA, nPowTwoLenA) + threadIdx.x;
    }

    __syncthreads();

    if (threadIdx.x < lenA)
    {
        dstKey[dstPosA] = keyA;
        dstVal[dstPosA] = valA;
    }

    if (threadIdx.x < lenB)
    {
        dstKey[dstPosB] = keyB;
        dstVal[dstPosB] = valB;
    }
}

extern "C"
__global__ void mergeElementaryIntervalsKernelUp(
    KEY_TYPE *d_DstKey,
    uint *d_DstVal,
    KEY_TYPE *d_SrcKey,
    uint *d_SrcVal,
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint stride,
    uint N
)
{
    __shared__ KEY_TYPE s_key[2 * SAMPLE_STRIDE];
    __shared__ uint s_val[2 * SAMPLE_STRIDE];

    const uint   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
    const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase;
    d_DstVal += segmentBase;

    //Set up threadblock-wide parameters
    __shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    if (threadIdx.x == 0)
    {
        uint segmentElementsA = stride;
        uint segmentElementsB = umin(stride, N - segmentBase - stride);
        uint  segmentSamplesA = getSampleCount(segmentElementsA);
        uint  segmentSamplesB = getSampleCount(segmentElementsB);
        uint   segmentSamples = segmentSamplesA + segmentSamplesB;

        startSrcA    = d_LimitsA[blockIdx.x];
        startSrcB    = d_LimitsB[blockIdx.x];
        uint endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        uint endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        if(endSrcA<startSrcA)
			lenSrcA=0;
		lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
        startDstA    = startSrcA + startSrcB;
        startDstB    = startDstA + lenSrcA;

    }

    //Load main input data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        s_key[threadIdx.x +             0] = d_SrcKey[0 + startSrcA + threadIdx.x];
        s_val[threadIdx.x +             0] = d_SrcVal[0 + startSrcA + threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x];
        s_val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
    }

    //Merge data in shared memory
    __syncthreads();
    merge<1>(
        s_key,
        s_val,
        s_key + 0,
        s_val + 0,
        s_key + SAMPLE_STRIDE,
        s_val + SAMPLE_STRIDE,
        lenSrcA, SAMPLE_STRIDE,
        lenSrcB, SAMPLE_STRIDE
    );

    //Store merged data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x];
        d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
        d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
    }
}


extern "C"
__global__ void mergeElementaryIntervalsKernelDown(
    KEY_TYPE *d_DstKey,
    uint *d_DstVal,
    KEY_TYPE *d_SrcKey,
    uint *d_SrcVal,
    uint *d_LimitsA,
    uint *d_LimitsB,
    uint stride,
    uint N
)
{
    __shared__ KEY_TYPE s_key[2 * SAMPLE_STRIDE];
    __shared__ uint s_val[2 * SAMPLE_STRIDE];

    const uint   intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
    const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
    d_SrcKey += segmentBase;
    d_SrcVal += segmentBase;
    d_DstKey += segmentBase;
    d_DstVal += segmentBase;

    //Set up threadblock-wide parameters
    __shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

    if (threadIdx.x == 0)
    {
        uint segmentElementsA = stride;
        uint segmentElementsB = umin(stride, N - segmentBase - stride);
        uint  segmentSamplesA = getSampleCount(segmentElementsA);
        uint  segmentSamplesB = getSampleCount(segmentElementsB);
        uint   segmentSamples = segmentSamplesA + segmentSamplesB;

        startSrcA    = d_LimitsA[blockIdx.x];
        startSrcB    = d_LimitsB[blockIdx.x];
        uint endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1] : segmentElementsA;
        uint endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1] : segmentElementsB;
        lenSrcA      = endSrcA - startSrcA;
        lenSrcB      = endSrcB - startSrcB;
        startDstA    = startSrcA + startSrcB;
        startDstB    = startDstA + lenSrcA;
    }

    //Load main input data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        s_key[threadIdx.x +             0] = d_SrcKey[0 + startSrcA + threadIdx.x];
        s_val[threadIdx.x +             0] = d_SrcVal[0 + startSrcA + threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        s_key[threadIdx.x + SAMPLE_STRIDE] = d_SrcKey[stride + startSrcB + threadIdx.x];
        s_val[threadIdx.x + SAMPLE_STRIDE] = d_SrcVal[stride + startSrcB + threadIdx.x];
    }

    //Merge data in shared memory
    __syncthreads();
    merge<0>(
        s_key,
        s_val,
        s_key + 0,
        s_val + 0,
        s_key + SAMPLE_STRIDE,
        s_val + SAMPLE_STRIDE,
        lenSrcA, SAMPLE_STRIDE,
        lenSrcB, SAMPLE_STRIDE
    );

    //Store merged data
    __syncthreads();

    if (threadIdx.x < lenSrcA)
    {
        d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x];
        d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
    }

    if (threadIdx.x < lenSrcB)
    {
        d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
        d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
    }
}

