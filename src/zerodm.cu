#include <fstream>
#include <iostream>
#include <string>

#include "filterbank.hpp"
#include "kernels.cuh"

#include "cufft.h"

using std::cout;
using std::endl;
using std::string;

int main(int argc, char *argv[])
{

    // NOTE: verbose mode ON for testing
    // TODO: make it a command line argument later
    bool verbose = true;

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



    // TODO: do stuff


    filfile.SaveFilterbank(outfile);

    return 0;
}
