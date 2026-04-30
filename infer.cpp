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
double midLayers[layerCount][cpl][T];
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

double outputFilter[outChannels][cpl];
double outputBiases[outChannels];

int head = T - 1;
int samplesSeen = 0;

void ReLU(double &x)
{
    if (x < 0)
        x = 0;
}

void buildIdx(int idx[k], int d)
{
    for (int i = 0; i < k; i++)
    {
        int delay = d * i;
        idx[i] = (delay < samplesSeen) ? (head - delay + T) & (T - 1) : -1; // bitmask wrap around works since 512 is a power of 2
    }
}

void dilatedConv(
    double *in,
    double *kernel,
    double &acc,
    const int idx[k])
{
    for (int i = 0; i < k; i++)
    {
        if (idx[i] < 0)
            break;
        acc += kernel[k - 1 - i] * in[idx[i]];
    }
}

void doInputLayer()
{
    int latest = head;
    int d = dilations[0];

    int idx[k];
    buildIdx(idx, d);

    // Conv1: input 6 -> 16
    for (int oc = 0; oc < cpl; oc++)
    {
        midLayers[0][oc][latest] = inputBias1[oc];

        for (int ic = 0; ic < inChannels; ic++)
            dilatedConv(input[ic], inputFilter1[oc][ic], midLayers[0][oc][latest], idx);

        ReLU(midLayers[0][oc][latest]);
    }

    // Conv2: 16 -> 16
    for (int oc = 0; oc < cpl; oc++)
    {
        hiddenLayers[0][oc][latest] = inputBias2[oc];

        for (int ic = 0; ic < cpl; ic++)
            dilatedConv(midLayers[0][ic], inputFilter2[oc][ic], hiddenLayers[0][oc][latest], idx);

        ReLU(hiddenLayers[0][oc][latest]);

        // Residual projection: input 6 -> 16
        double res = inputResidualBias[oc];

        for (int ic = 0; ic < inChannels; ic++)
            res += inputResidualFilter[oc][ic] * input[ic][latest];

        hiddenLayers[0][oc][latest] += res;
    }
}

void doHiddenLayer(int layerNum)
{
    int latest = head;
    int d = dilations[layerNum];
    int f = layerNum - 1;

    int idx[k];
    buildIdx(idx, d);

    // Conv1: 16 -> 16
    for (int oc = 0; oc < cpl; oc++)
    {
        midLayers[layerNum][oc][latest] = hiddenBias1[f][oc];

        for (int ic = 0; ic < cpl; ic++)
            dilatedConv(hiddenLayers[layerNum - 1][ic], hiddenFilter1[f][oc][ic], midLayers[layerNum][oc][latest], idx);

        ReLU(midLayers[layerNum][oc][latest]);
    }

    // Conv2: 16 -> 16
    for (int oc = 0; oc < cpl; oc++)
    {
        hiddenLayers[layerNum][oc][latest] = hiddenBias2[f][oc];

        for (int ic = 0; ic < cpl; ic++)
            dilatedConv(midLayers[layerNum][ic], hiddenFilter2[f][oc][ic], hiddenLayers[layerNum][oc][latest], idx);

        ReLU(hiddenLayers[layerNum][oc][latest]);

        // Identity residual: 16 -> 16
        hiddenLayers[layerNum][oc][latest] += hiddenLayers[layerNum - 1][oc][latest];
    }
}

void doOutputLayer(double output[outChannels])
{
    int latest = head;

    for (int oc = 0; oc < outChannels; oc++)
    {
        output[oc] = outputBiases[oc];

        for (int ic = 0; ic < cpl; ic++)
            output[oc] += outputFilter[oc][ic] * hiddenLayers[layerCount - 1][ic][latest];
    }
}

void shiftInput(double newSample[inChannels])
{
    head = (head + 1) & (T - 1);

    if (samplesSeen < T)
        samplesSeen++;

    for (int ch = 0; ch < inChannels; ch++)
        input[ch][head] = newSample[ch];
}

void inferNext(double newSample[inChannels], double prediction[outChannels])
{
    shiftInput(newSample);
    doInputLayer();

    for (int layer = 1; layer < layerCount; layer++)
        doHiddenLayer(layer);

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