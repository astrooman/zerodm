#ifndef _H_ZERODM_KERNELS
#define _H_ZERODM_KERNELS

#include "cuda.h"

__global__ void GetMeansKernel(unsigned char *indata, float *means, unsigned int nchans, unsigned int perthread);

__global__ void RemoveZeroDmKernel(unsigned char *indata, float *means, unsigned int nchans, unsigned int perthread);

#endif
