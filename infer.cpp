#include <iostream>
using namespace std;

void ReLU(double * in, int len) {
    for(int i = 0; i < len; i++) {
        in[i] = (in[i] > 0) ? in[i] : 0;
    }
}

double dilatedConv(double * in, double * kernel, double b, int inLen, int k, int d) {
    // in - input sequence, format : last entry is the latest entry
    // kernel - filter, format : same as the input, last weight is for latest entry
    // b - bias
    // inLen - input sequence length
    // k - filter length
    // d - dilation factor
    double out = b;

    for(int i = 0; i < k; i++) {
        if(d * i > inLen - 1) continue; // zero padding for when we go beyond the size of the sequnce
        out += kernel[k - 1 - i] * in[inLen - (1 + d*i)]; // this skips every point according to the dilation factor
    }

    return out;
}

double * doHiddenLayer(double * in, int inLen, double * kernel, double * b, int inChannels, int k, int outChannels, int d, double * residual) {
    // This function does the dilated convolution + ReLU + residual for a single hidden layer (or the output layer)
    // This function returns a 1d array of length kWidth composed of the latest outputs in a single layer

    // in - input sequence, can be a 2d array assuming multiple channels. this can be the actual input or a previously computed hidden layer
    // inLen - this is the length of the input layer
    // kernel - this is the filter, can be 3d if the input sequence has multiple channels
    // b - this is an array of biases for each channel, should be 1d
    // k - this is the length of one individual kernal
    // inChannels - this is the length of the kernel, to be passed into the dilatedConv function
    // outChannels - this is the width/height of the kernel, this is how we determine the number of input channels
    // residual - this is added to each point, can be an array
    double* out = new double[outChannels];
    for(int i = 0; i < outChannels; i++) {
        out[i] = b[i];
        for(int j = 0; j < inChannels; j++) {
            out[i] += dilatedConv(&in[j * inLen], &kernel[(i * inChannels + j) * k], 0, inLen, k, d);
        }
    }
    ReLU(out, outChannels);
    for(int i = 0; i < outChannels; i++ ) {
        out[i] += residual[i];
    }

    return out;
}