#ifndef _CONVOLUTIONAL_PADDED_HPP_
#define _CONVOLUTIONAL_PADDED_HPP_

#include "ActivationFunction.hpp"
#include "Layer.hpp"
#include "UpdateRule.hpp"
#include "fft2D.hpp"//自己添加的头文件，用于各种二维DCT的实现

#include <fftw3.h>

class ConvolutionalPadded: public Layer
{
	public:
		ConvolutionalPadded(std::size_t rank,const std::size_t *imageDims, const std::size_t *kernelDims, 
			std::size_t inputs, std::size_t outputs, nnet_float initweight, ActivationFunction *func, UpdateRule *ur,std::size_t P);
		~ConvolutionalPadded();
		void startBatch() override;
		void endBatch() override;
		void load(std::istream &is) override;
		void forward(const nnet_float *features) override;
		void backward(nnet_float *bpDeltaErrors) override;
		void calculateGradients(const nnet_float *features) override;
		void updateWeights(const unsigned int batchSize) override;
		void updateBiases(const unsigned int batchSize) override;
		std::string toString() const override;

	protected:
		std::size_t numInputChannels;
		std::size_t numOutputChannels;
		std::size_t kernelVolume;
		std::size_t frequencyVolume;
		std::size_t inputVolume;
		std::size_t outputVolume;
		std::size_t featureVolume;
		nnet_float *padded;
		nnet_float *frequencyActivations;
		nnet_float *frequencyWeights;
		nnet_float *frequencyInputs;
		nnet_float *frequencyDeltaErrors;
		nnet_float *frequencyDeltaWeights;
		fftwf_plan forwardTransform;//声明一个plan，用于执行DFT
		fftwf_plan backwardTransform;//用自己的实现，替换
		std::size_t tensorRank;
		std::size_t *inputDimensions;
		std::size_t *kernelDimensions;
		std::size_t *outputDimensions;
		std::size_t *paddedInputDimensions;  //pad input image
		nnet_float initWeight;
		ActivationFunction *activationFunction;
		UpdateRule *updateRule;
		
};

#endif
