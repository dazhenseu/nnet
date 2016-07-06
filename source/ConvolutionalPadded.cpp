#include <cstring>
#include <sstream>

#include "ConvolutionalPadded.hpp"
#include "vector.hpp"
#include "core.hpp"
#include "fftw3.h"

using namespace std;

ConvolutionalPadded::ConvolutionalPadded(size_t rank, const size_t *inputDims, const size_t *kernelDims, 
				size_t inputs, size_t outputs, nnet_float initweight, ActivationFunction *func, UpdateRule *ur,std::size_t P=0)
{
	numInputChannels = inputs;
	numOutputChannels = outputs;
	numInputs = inputs;
	numOutputs = outputs;
	numWeights = inputs * outputs;
	numBiases = outputs;
	inputVolume = 1;
	outputVolume = 1;
	kernelVolume = 1;
	frequencyVolume = 2;
	outputDimensions = new size_t[rank];
	inputDimensions = new size_t[rank];
	kernelDimensions = new size_t[rank];
	paddedInputDimensions = new size_t [rank]; //to record padded size of input images
	memcpy((void *)inputDimensions, inputDims, sizeof(size_t) * rank);
	memcpy((void *)kernelDimensions, kernelDims, sizeof(size_t) * rank);

	for(size_t i = 0; i < rank; i++)
	{
		paddedInputDimensions[i] = inputDimensions[i]+ 2*P;
		numInputs *= (inputDims[i]+2*P);
		numOutputs *= (inputDims[i] + 2*P - kernelDims[i] + 1);
		outputDimensions[i] = inputDims[i] + 2*P - kernelDims[i] + 1;
		numWeights *= kernelDims[i];
		inputVolume *= (inputDims[i]+ 2*P);
		featureVolume *=inputDims[i];//real input feature volume
		frequencyVolume *= (inputDims[i]+ 2*P);
		kernelVolume *= kernelDims[i];
		outputVolume *= (inputDims[i] + 2*P - kernelDims[i] + 1);
	}

	if(rank > 0)
	{
		frequencyVolume = (frequencyVolume / paddedInputDimensions[rank - 1]) * (paddedInputDimensions[rank - 1] / 2 + 1);
	}

	tensorRank = rank;
	initWeight = initweight;//0.01
	activationFunction = func;
	updateRule = ur;
		
	
	weights = nnet_malloc(weightsSize());
	deltaWeights = nnet_malloc(weightsSize());
	biases = nnet_malloc(biasesSize());
	deltaBiases = nnet_malloc(biasesSize());
	activations = nnet_malloc(outputsSize());
	deltaActivations = nnet_malloc(outputsSize());
	deltaErrors = nnet_malloc(outputsSize());
	weightsMomentum = nnet_malloc(weightsSize());
	biasesMomentum = nnet_malloc(biasesSize());

	padded = nnet_malloc(inputVolume);
	frequencyActivations = nnet_malloc(frequencyVolume);
	frequencyWeights = nnet_malloc(frequencyVolume * inputs * outputs);
	frequencyInputs = nnet_malloc(frequencyVolume * inputs);
	frequencyDeltaErrors = nnet_malloc(frequencyVolume * outputs);
	frequencyDeltaWeights = nnet_malloc(frequencyVolume * inputs * outputs);

	//Initialise the weights
	random_gaussian_vector(weights, numWeights, 0.0, initWeight);

	//Initialise the biases
	random_gaussian_vector(biases, numBiases, 1.0, initWeight);

	int *dims = new int[tensorRank];

	for(size_t i = 0; i < tensorRank; i++)
	{
		dims[i] = paddedInputDimensions[i];
	}
	
	forwardTransform = fftwf_plan_dft_r2c(tensorRank, dims, padded, (fftwf_complex *)frequencyInputs, FFTW_EXHAUSTIVE);//fftwf_plan fftwf_plan_dft_r2c(int rank, const int *n, float *in,fftwf_complex 
 															   //*out, unsigned flags);
	backwardTransform = fftwf_plan_dft_c2r(tensorRank, dims, (fftwf_complex *)frequencyInputs, padded, FFTW_EXHAUSTIVE);

	delete[] dims;

	memset(deltaWeights, 0, sizeof(nnet_float) * numWeights);
	memset(deltaBiases, 0, sizeof(nnet_float) * numBiases);
	memset(weightsMomentum, 0, sizeof(nnet_float) * numWeights);
	memset(biasesMomentum, 0, sizeof(nnet_float) * numBiases);
}

ConvolutionalPadded::~ConvolutionalPadded()
{
	nnet_free(weights);
	nnet_free(deltaWeights);
	nnet_free(biases);
	nnet_free(deltaBiases);
	nnet_free(activations);
	nnet_free(deltaActivations);
	nnet_free(deltaErrors);
	nnet_free(weightsMomentum);
	nnet_free(biasesMomentum);

	nnet_free(padded);
	nnet_free(frequencyActivations);
	nnet_free(frequencyWeights);
	nnet_free(frequencyInputs);
	nnet_free(frequencyDeltaErrors);
	nnet_free(frequencyDeltaWeights);
	delete []paddedInputDimensions;
}

void ConvolutionalPadded::load(istream &is)
{
	Layer::load(is);

	//Convert the filters into the frequency domain
	startBatch();
}

void ConvolutionalPadded::startBatch()
{
	nnet_float *ws = weights;
	nnet_float *fws = frequencyWeights;
	nnet_float *fdws = frequencyDeltaWeights;
	
	
	for(size_t i = 0; i < numInputChannels * numOutputChannels; i++)
	{	
		
		pad(tensorRank, ws, kernelDimensions, padded, paddedInputDimensions);
		fftwf_execute_dft_r2c(forwardTransform, padded, (fftwf_complex *)fws);
		memset(fdws, 0, sizeof(nnet_float) * frequencyVolume);
		ws += kernelVolume;
		fws += frequencyVolume;
		fdws += frequencyVolume;
	}
}

void ConvolutionalPadded::endBatch()
{
	nnet_float normaliser = 1.0 / (nnet_float)inputVolume;

	nnet_float *fdws = frequencyDeltaWeights;
	nnet_float *dws = deltaWeights;
	
	for(size_t i = 0; i < numInputChannels * numOutputChannels; i++)
	{
		fftwf_execute_dft_c2r(backwardTransform, (fftwf_complex *)fdws, padded);//，void fftw_execute_dft_c2r(const fftw_plan p,fftw_complex *in, double *out);
		extract_valid_rotate(tensorRank, padded, paddedInputDimensions, dws, kernelDimensions, normaliser);
		fdws += frequencyVolume;
		dws += kernelVolume;
	}
}

void ConvolutionalPadded::forward(const nnet_float *features)
{
	nnet_float normaliser = 1.0 / (nnet_float)inputVolume;
	nnet_float *ffs = frequencyInputs;
	nnet_float *fas = frequencyActivations;
	nnet_float *fws = frequencyWeights;
	nnet_float *as = activations;
	
	//Transform all the input channels into the frequency domain
	for(size_t i = 0; i < numInputChannels; i++)
	{	
		pad(tensorRank, features, inputDimensions, padded, paddedInputDimensions);
		fftwf_execute_dft_r2c(forwardTransform, padded, (fftwf_complex *)ffs);
		features += featureVolume;
		//features += inputVolume;
		ffs += frequencyVolume;
	}

	for(size_t i = 0; i < numOutputChannels; i++)
	{
		ffs = frequencyInputs;
		memset(fas, 0, sizeof(nnet_float) * frequencyVolume);
		for(size_t j = 0; j < numInputChannels; j++)
		{
			vector_complex_fma(fas, fws, ffs, frequencyVolume / 2);
			fws += frequencyVolume;
			ffs += frequencyVolume;
		}

		fas[0] += biases[i] * (nnet_float)inputVolume;

		fftwf_execute_dft_c2r(backwardTransform, (fftwf_complex *)fas, padded);
		extract_valid(tensorRank, padded, paddedInputDimensions, as, outputDimensions);
		vector_scale(as, outputVolume, normaliser);
		as += outputVolume;
	}

	(*activationFunction)(activations, deltaActivations, numOutputs);
}

void ConvolutionalPadded::backward(nnet_float *bpDeltaErrors)
{
	nnet_float *fdes = frequencyDeltaErrors;
	nnet_float *fws;
	nnet_float *ftemp = frequencyActivations;
	nnet_float normalisation = 1.0 / (nnet_float)inputVolume;

	//Transform deltaError maps into the frequency domain
	/*for(size_t i = 0; i < numOutputChannels; i++)
	{
		pad_rotate(tensorRank, des, outputDimensions, padded, inputDimensions);
		fftwf_execute_dft_r2c(forwardTransform, padded, (fftwf_complex *)fdes);

		des += outputVolume;
		fdes += frequencyVolume;
	}*/

	for(size_t i = 0; i < numInputChannels; i++)
	{
		memset(ftemp, 0, sizeof(nnet_float) * frequencyVolume);
		fws = frequencyWeights + i * frequencyVolume;
		fdes = frequencyDeltaErrors;

		for(size_t j = 0; j < numOutputChannels; j++)
		{
			vector_complex_fma(ftemp, fws, fdes, frequencyVolume / 2);
			fws += frequencyVolume * numInputChannels;
			fdes += frequencyVolume;
		}

		fftwf_execute_dft_c2r(backwardTransform, (fftwf_complex *)ftemp, padded);
		extract_full_rotate(tensorRank, padded, paddedInputDimensions, bpDeltaErrors, inputDimensions);
		vector_scale(bpDeltaErrors, inputVolume, normalisation);
		bpDeltaErrors += featureVolume;
	}
}

void ConvolutionalPadded::calculateGradients(const nnet_float *features)
{
	nnet_float *fdws = frequencyDeltaWeights;
	nnet_float *fdes = frequencyDeltaErrors;
	nnet_float *des = deltaErrors;
	nnet_float *ffs;

	vector_mul(deltaErrors, deltaActivations, deltaErrors, numOutputs);

	for(size_t i = 0; i < numOutputChannels; i++)
	{
		ffs = frequencyInputs;
		deltaBiases[i] += vector_sum(des, outputVolume);

		pad_rotate(tensorRank, des, outputDimensions, padded, paddedInputDimensions);
		fftwf_execute_dft_r2c(forwardTransform, padded, (fftwf_complex *)fdes);

		for(size_t j = 0; j < numInputChannels; j++)
		{
			vector_complex_fma(fdws, fdes, ffs, frequencyVolume / 2);
			ffs += frequencyVolume;
			fdws += frequencyVolume;
		}

		des += outputVolume;
		fdes += frequencyVolume;
	}	
}

void ConvolutionalPadded::updateWeights(const unsigned int batchSize)
{
	updateRule->updateWeights(this, batchSize);
}

void ConvolutionalPadded::updateBiases(const unsigned int batchSize)
{
	updateRule->updateBiases(this, batchSize);
}

string ConvolutionalPadded::toString() const
{
	stringstream output;

	output << "ConvolutionalPadded\n"
		<< "\tInput Channels: " << numInputChannels << "\n"
		<< "\tOutput Channels: " << numOutputChannels << "\n";

	return output.str();
}
