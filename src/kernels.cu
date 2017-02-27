#include "cuda.h"

#include "kernels.cuh"

__global__ void GetMeansKernel(unsigned char *indata, float *means, unsigned int nchans, unsigned int perthread) {

    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int startidx = 0;

    for (unsigned int isamp = 0; isamp < perthread; isamp++) {
        means[perthread * xidx + isamp] = 0.0f;
        startidx = perthread * xidx * nchans + isamp * nchans;
        for (unsigned int ichan = 0; ichan < nchans; ichan++) {
            means[perthread * xidx + isamp] += indata[startidx + ichan];
        }
    }
}

__global__ void RemoveZeroDmKernel(unsigned char *indata, float *means, unsigned int nchans, unsigned int perthread) {

    int xidx = blockDim.x * blockDim.x + threadIdx.x;
    int startidx = 0;

    float fdiff = 0.0f;

    for (unsigned int isamp = 0; isamp < perthread; isamp++) {
        startidx = perthread * xidx * nchans + isamp * nchans;
        for (unsigned int ichan = 0; ichan < nchans; ichan++) {
            fdiff = indata[startidx + ichan] - means[perthread * xidx + isamp];
            // NOTE: rintf() rounds the halfway cases towards zero - rounding approach will most probably change
            // NOTE: new mean is set to 64
            indata[startidx + ichan] = rintf(fdiff) + 64;
        }
    }

}
