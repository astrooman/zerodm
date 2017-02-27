#include <fstream>
#include <iostream>
#include <string>

#include "cuda.h"

#include "errors.hpp"
#include "filterbank.hpp"
#include "kernels.cuh"

using std::cout;
using std::endl;
using std::string;

int main(int argc, char *argv[])
{

    // NOTE: verbose mode ON for testing
    // TODO: make it a command line argument later
    bool verbose = true;

    // TODO: include proper command line argument parsing
    string infile, outfile;
    infile = argv[1];

    if (argv[2] == NULL) {
        outfile = infile;
    } else {
        outfile = argv[2];
    }

    Filterbank filfile(infile);
    FilHeader filhead = filfile.GetHeader();

    size_t totalsamples = filhead.nchans * filhead.nsamps;
    size_t totalbytes = totalsamples * filhead.nbits / 8;

    if (verbose) {
        cout << "Channels: " << filhead.nchans << endl;
        cout << "Time samples per channel: " << filhead.nsamps << endl;
        cout << "Bits per sample: " << filhead.nbits << endl;
    }

    size_t chunksamples = 1<<18;
    // full chunks first
    // worry about the remainder later
    unsigned int nochunks = filhead.nsamps / chunksamples;

    // number of bytes to read for every full chunk
    size_t toread = chunksamples * filhead.nchans * nbits / 8;
    unsigned char *dchunk;
    cudaCheckError(cudaMalloc((void**)&dchunk, toread * sizeof(unsigned char)));

    float *dmeans;
    cudaCheckError(cudaMalloc((void**)&dmeans, chunksamples * sizeof(float)));

    // NOTE: make sure this is a power of 2
    unsigned int perthread = 1;

    dim3 nthreads(1024, 1, 1);
    dim3 nblocks(chunksamples / nthreads.x / perthread, 1, 1);

    for (unsigned int ichunk = 0; ichunk < nochunks; ichunk++) {

        cudaCheckError(cudaMemcpy(dchunk, filfile.GetFilData() + ichunk * toread, toread * sizeof(unsigned char), cudaMemcpyHostToDevice));

        GetMeansKernel<<<nblocks, nthreads, 0, 0>>>(dchunk, dmeans, filhead.nchans, perthread);
        RemoveZeroDmKernel<<<nblocks, nthreads, 0, 0>>>(dchunk, dmeans, filchead.nchans, perthread);

        cudaCheckError(cudaMemcpy(filfile.GetFilData() + ichunk * toread, dchunk, toread * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        cudaCheckError(cudaDeviceSynchronize());
    }

    cudaCheckError(cudaFree(dmeans));
    cudaCheckError(cudaFree(dchunk));

    filfile.SaveFilterbank(outfile);

    return 0;
}
