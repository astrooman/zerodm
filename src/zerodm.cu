#include <fstream>
#include <iomanip>
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
    size_t remainder = filhead.nsamps - nochunks * chunksamples;

    cout << "Remainder " << remainder << endl;

    // number of bytes to read for every full chunk
    size_t toread = chunksamples * filhead.nchans * filhead.nbits / 8;

    if (verbose) {
        cout << "Total file size: " << (float)totalbytes / 1024.0 / 1024.0 << "MB\n";
        cout << "Chunks size: " << (float)toread / 1024.0 / 1024.0 << "MB\n";
    }

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

        GetMeansKernel<<<nblocks, nthreads, 0, 0>>>(dchunk, dmeans, filhead.nchans, chunksamples, perthread);
        cudaCheckError(cudaDeviceSynchronize());
        RemoveZeroDmKernel<<<nblocks, nthreads, 0, 0>>>(dchunk, dmeans, filhead.nchans, chunksamples, perthread);
        cudaCheckError(cudaDeviceSynchronize());

        cudaCheckError(cudaMemcpy(filfile.GetFilData() + ichunk * toread, dchunk, toread * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	cout << "\rFinished " << std::fixed << std::setprecision(2) << (float)(ichunk + 1) / (float)(nochunks + (remainder != 0)) * 100.0f << "%";
        cout.flush();
    }

    if (remainder != 0) {
     
        cudaCheckError(cudaMemcpy(dchunk, filfile.GetFilData() + nochunks * toread, remainder * filhead.nchans * filhead.nbits / 8 * sizeof(unsigned char), cudaMemcpyHostToDevice));
        
        GetMeansKernel<<<nblocks, nthreads, 0, 0>>>(dchunk, dmeans, filhead.nchans, remainder, 1);
        cudaCheckError(cudaDeviceSynchronize());
        RemoveZeroDmKernel<<<nblocks, nthreads, 0, 0>>>(dchunk, dmeans, filhead.nchans, remainder, 1);
        cudaCheckError(cudaDeviceSynchronize());

        cudaCheckError(cudaMemcpy(filfile.GetFilData() + nochunks * toread, dchunk, remainder * filhead.nchans * filhead.nbits / 8 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    
        cout << "\rFinished 100.00%";

    }

    cout << endl;

    cudaCheckError(cudaFree(dmeans));
    cudaCheckError(cudaFree(dchunk));

    filfile.SaveFilterbank(outfile);

    return 0;
}
