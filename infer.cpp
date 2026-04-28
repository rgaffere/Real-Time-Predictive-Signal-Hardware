#include <iostream>
using namespace std;

// Since we're focused on inference, we only do ReLU on the latest point
void ReLU(double &x)
{
    if (x < 0)
        x = 0;
}

void dilatedConv(double *in, double *kernel, double &acc, int T, int k, int d)
{
    // in - input sequence, format : last entry is the latest entry
    // kernel - filter, format : same as the input, last weight is for latest entry
    // acc - accumulator, pass by reference so we dont have an output. cleaner solution
    // T - input sequence length
    // k - filter length
    // d - dilation factor
    for (int i = 0; i < k; i++)
    {
        int index = T - (1 + d * i);
        if (index < 0)
            break;                            // zero padding for when we go beyond the size of the sequnce
        acc += kernel[k - 1 - i] * in[index]; // this skips every point according to the dilation factor
    }
}

void doResiBlock(double *in, double *h1, double *b1, double *k1, double *out, double *b2, double *k2, double *k3, double *b3, int k, int d, int T, int inChannels, int outChannels)
{
    // in - input sequence, could be the output of a hidden layer, or the very first layer. depending on channel count is either 1d or 2d
    // h1 - result of first convolution, can be either 1d or 2d depending on channel count
    // out - result of second convolution, can be 1d or 2d
    // b1 - this is a 1d array of biases to be added to in for the computation of h1
    // k1 - this is the filter to be applied to in for the computation of h1
    // out - this is the result of the second convolution, can be 1d or 2d
    // b2 - this is a 1d array of biases to be applied to h1 in order to compute out
    // k2 - this is the filter to be applied onto h1 in order to compute out
    // k3,b3 - this is for the residual addition.
    // k - this is the filter length
    // d - this is the dilation factor
    // T - this is the length of the input sequence and h1 and out
    // inChannels - this is the number of input channels (applies to in)
    // outChannels - this is the number of output channels (applies to h1 and out)

    // first convolution
    int latest = T - 1;
    for (int i = 0; i < outChannels; i++)
    {
        h1[T * i + latest] = b1[i];

        for (int j = 0; j < inChannels; j++)
        {
            dilatedConv(&in[j * T], &k1[(i * inChannels + j) * k], h1[T * i + latest], T, k, d);
        }
        ReLU(h1[T * i + latest]);
    }
    // second convolution
    for (int i = 0; i < outChannels; i++)
    {
        out[T * i + latest] = b2[i];

        for (int j = 0; j < outChannels; j++)
        {
            dilatedConv(&h1[j * T], &k2[(i * outChannels + j) * k], out[T * i + latest], T, k, d);
        }
        ReLU(out[T * i + latest]);
    }
    // now do the residual add, if in and out channels are the same, k3 is an identity matrix
    for (int oc = 0; oc < outChannels; oc++)
    {
        double res = b3[oc];

        for (int ic = 0; ic < inChannels; ic++)
        {
            res += k3[oc * inChannels + ic] * in[ic * T + latest];
        }

        out[oc * T + latest] += res;
    }
}
