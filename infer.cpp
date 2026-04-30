/*
Model Architecture

Input: 6 channels
Layers: 7
Channels per layer: 16
Kernel size: 3
Dilation: 1,2,4,8,16,32,64
Output: 6 channels (prediction)
Mode: streaming (latest point only)
*/

const int inChannels = 6;
const int layerCount = 7;
const int cpl = 16;
const int k = 3;
const int dilations[layerCount] = {1, 2, 4, 8, 16, 32, 64};
const int outChannels = 6;
const int T = 512; // chosen to cover RF = 1 + 2 * (k - 1) * sum(dilations) = 509

double input[inChannels][T];
double midLayers[layerCount][cpl][T]; // we need this because we do two convolutions
double hiddenLayers[layerCount][cpl][T];

double inputFilter1[cpl][inChannels][k];
double inputBias1[cpl];

double inputFilter2[cpl][cpl][k];
double inputBias2[cpl];

double inputResidualFilter[cpl][inChannels];
double inputResidualBias[cpl];

double hiddenFilter1[layerCount - 1][cpl][cpl][k];
double hiddenBias1[layerCount - 1][cpl];

double hiddenFilter2[layerCount - 1][cpl][cpl][k];
double hiddenBias2[layerCount - 1][cpl];

// no need to cache the output layer since its just inference for the latest point
double outputFilter[outChannels][cpl];
double outputBiases[outChannels];

void ReLU(
    double &x // Since we're focused on inference, we only do ReLU on the latest point
)
{
    if (x < 0)
        x = 0;
}

void dilatedConv(
    double *in,     // in - input sequence, format : last entry is the latest entry
    double *kernel, // kernel - filter, format : same as the input, last weight is for latest entry
    double &acc,    // acc - accumulator, pass by reference so we dont have an output. cleaner solution
    int d)
{
    for (int i = 0; i < k; i++)
    {
        int index = T - (1 + d * i);
        if (index < 0)
            break;                            // zero padding for when we go beyond the size of the sequnce
        acc += kernel[k - 1 - i] * in[index]; // this skips every point according to the dilation factor
    }
}

void doInputLayer()
{
    int latest = T - 1;
    int d = dilations[0];

    // Conv1: input 6 -> 16
    for (int oc = 0; oc < cpl; oc++)
    {
        double acc = inputBias1[oc];

        for (int ic = 0; ic < inChannels; ic++)
            dilatedConv(input[ic], inputFilter1[oc][ic], acc, d);

        ReLU(acc);
        midLayers[0][oc][latest] = acc;
    }

    // Conv2: 16 -> 16
    for (int oc = 0; oc < cpl; oc++)
    {
        double acc = inputBias2[oc];

        for (int ic = 0; ic < cpl; ic++)
            dilatedConv(midLayers[0][ic], inputFilter2[oc][ic], acc, d);

        ReLU(acc);

        // Residual projection: input 6 -> 16
        double res = inputResidualBias[oc];

        for (int ic = 0; ic < inChannels; ic++)
            res += inputResidualFilter[oc][ic] * input[ic][latest];

        hiddenLayers[0][oc][latest] = acc + res;
    }
}

void doHiddenLayer(int layerNum)
{
    // layerNum goes from 1 to 6
    int latest = T - 1;
    int d = dilations[layerNum];
    int f = layerNum - 1;

    // Conv1: 16 -> 16
    for (int oc = 0; oc < cpl; oc++)
    {
        double acc = hiddenBias1[f][oc];

        for (int ic = 0; ic < cpl; ic++)
            dilatedConv(hiddenLayers[layerNum - 1][ic], hiddenFilter1[f][oc][ic], acc, d);

        ReLU(acc);
        midLayers[layerNum][oc][latest] = acc;
    }

    // Conv2: 16 -> 16
    for (int oc = 0; oc < cpl; oc++)
    {
        double acc = hiddenBias2[f][oc];

        for (int ic = 0; ic < cpl; ic++)
            dilatedConv(midLayers[layerNum][ic], hiddenFilter2[f][oc][ic], acc, d);

        ReLU(acc);

        // Identity residual: 16 -> 16
        hiddenLayers[layerNum][oc][latest] =
            acc + hiddenLayers[layerNum - 1][oc][latest];
    }
}

void doOutputLayer(double output[outChannels])
{
    int latest = T - 1;

    for (int oc = 0; oc < outChannels; oc++)
    {
        double acc = outputBiases[oc];

        for (int ic = 0; ic < cpl; ic++)
            acc += outputFilter[oc][ic] * hiddenLayers[layerCount - 1][ic][latest];

        output[oc] = acc;
    }
}

// now for some cache handling
void shiftInput(double newSample[inChannels])
{
    for (int ch = 0; ch < inChannels; ch++)
    {
        for (int t = 0; t < T - 1; t++)
            input[ch][t] = input[ch][t + 1];

        input[ch][T - 1] = newSample[ch];
    }
}

void shiftHiddenState()
{
    for (int layer = 0; layer < layerCount; layer++)
    {
        for (int ch = 0; ch < cpl; ch++)
        {
            for (int t = 0; t < T - 1; t++)
            {
                midLayers[layer][ch][t] = midLayers[layer][ch][t + 1];
                hiddenLayers[layer][ch][t] = hiddenLayers[layer][ch][t + 1];
            }

            midLayers[layer][ch][T - 1] = 0.0;
            hiddenLayers[layer][ch][T - 1] = 0.0;
        }
    }
}

void inferNext(double newSample[inChannels], double prediction[outChannels])
{
    // 1. Shift cached input/history state
    shiftInput(newSample);
    shiftHiddenState();

    // 2. Compute newest timestep through the TCN
    doInputLayer();

    for (int layer = 1; layer < layerCount; layer++)
        doHiddenLayer(layer);

    // 3. Project final hidden state to predicted next IMU sample
    doOutputLayer(prediction);
}

double computeError(double prediction[outChannels], double actual[outChannels])
{
    double err = 0.0;

    for (int i = 0; i < outChannels; i++)
    {
        double diff = actual[i] - prediction[i];
        err += diff * diff;
    }

    return err;
}