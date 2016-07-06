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
	numWeights = inputs * outputs;//1*112，这个numWeights成员变量继承自Layer类的保护成员，可以访问
	numBiases = outputs;
	inputVolume = 1;
	outputVolume = 1;
	kernelVolume = 1;
	frequencyVolume = 2;//因为有实部和虚部都是放在了一个数组中
	outputDimensions = new size_t[rank];//平面图像，rank表示维数为
	inputDimensions = new size_t[rank];
	kernelDimensions = new size_t[rank];
	paddedInputDimensions = new size_t [rank];
	memcpy((void *)inputDimensions, inputDims, sizeof(size_t) * rank);//从inputDims拷贝sizeof(size_t) * rank个字节到inputDimensions所指的内存中
	memcpy((void *)kernelDimensions, kernelDims, sizeof(size_t) * rank);

	for(size_t i = 0; i < rank; i++)
	{
		paddedInputDimensions[i] = inputDimensions[i]+ 2*P;
		numInputs *= (inputDims[i]+2*P);//28=1*28,28*28=784, add pad pixels
		numOutputs *= (inputDims[i] + 2*P - kernelDims[i] + 1);//numOutputs:2240=112*20,2240*20=44800
		outputDimensions[i] = inputDims[i] + 2*P - kernelDims[i] + 1;//20,20
		numWeights *= kernelDims[i];//l1-l2层之间的权值个数有112*9=1008,1008*9=9072
		inputVolume *= (inputDims[i]+ 2*P);//inputVolume:28*28=784
		featureVolume *=inputDims[i];//real input feature volume
		frequencyVolume *= (inputDims[i]+ 2*P);//frequencyVolume:2*28=56,56*28=1568
		kernelVolume *= kernelDims[i];//9*9=81
		outputVolume *= (inputDims[i] + 2*P - kernelDims[i] + 1);//20*20=400
	}

	if(rank > 0)//we use the term rank to denote the number of independent indices in an array.
	{
		frequencyVolume = (frequencyVolume / paddedInputDimensions[rank - 1]) * (paddedInputDimensions[rank - 1] / 2 + 1);//rank=2,frequencyVolume=56×（14+1）=840，
									//这里frequencyVolume表示的是一副图像进行FFT变换后实际所占大小，由于输入是实数，利用了埃尔米特对称性。
	}

	tensorRank = rank;
	initWeight = initweight;//0.01
	activationFunction = func;//选用激活函数
	updateRule = ur;
		
	//下面是要计算的变量，都是从Layer类继承过来的
	weights = nnet_malloc(weightsSize());//weightsSize是Layer类的一个虚函数，return numWeights
	deltaWeights = nnet_malloc(weightsSize());
	biases = nnet_malloc(biasesSize());//biasesSize也是Layer类的一个虚函数，return numBiases;
	deltaBiases = nnet_malloc(biasesSize());
	activations = nnet_malloc(outputsSize());//outputSize()函数return numOutputs;
	deltaActivations = nnet_malloc(outputsSize());
	deltaErrors = nnet_malloc(outputsSize());
	weightsMomentum = nnet_malloc(weightsSize());
	biasesMomentum = nnet_malloc(biasesSize());

	padded = nnet_malloc(inputVolume);//属于该类的一个指针变量，注意到这里的padded的尺寸为784=28*28，实际用于填充后的卷积核,also used to padded input image 
	frequencyActivations = nnet_malloc(frequencyVolume);//频域中的激活值？
	frequencyWeights = nnet_malloc(frequencyVolume * inputs * outputs);//840*1*112
	frequencyInputs = nnet_malloc(frequencyVolume * inputs);
	frequencyDeltaErrors = nnet_malloc(frequencyVolume * outputs);
	frequencyDeltaWeights = nnet_malloc(frequencyVolume * inputs * outputs);

	//Initialise the weights
	random_gaussian_vector(weights, numWeights, 0.0, initWeight);//实现在vector.cpp中，均值为0，标准差为0.01

	//Initialise the biases
	random_gaussian_vector(biases, numBiases, 1.0, initWeight);

	int *dims = new int[tensorRank];//tensorRank的值为2

	for(size_t i = 0; i < tensorRank; i++)
	{
		dims[i] = paddedInputDimensions[i];
	}
	//可以替换的部分
	forwardTransform = fftwf_plan_dft_r2c(tensorRank, dims, padded, (fftwf_complex *)frequencyInputs, FFTW_EXHAUSTIVE);//fftwf_plan fftwf_plan_dft_r2c(int rank, const int *n, float *in,fftwf_complex 
 															   //*out, unsigned flags);返回这样设置的一个plan
	backwardTransform = fftwf_plan_dft_c2r(tensorRank, dims, (fftwf_complex *)frequencyInputs, padded, FFTW_EXHAUSTIVE);

	delete[] dims;

	memset(deltaWeights, 0, sizeof(nnet_float) * numWeights);//初始化deltaWeights的值为0，并返回指针deltaWeights下同
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
	nnet_float *ws = weights;//由load函数得到的weights
	nnet_float *fws = frequencyWeights;//frequencyWeights在构造函数中已经分配了内存
	nnet_float *fdws = frequencyDeltaWeights;
	
	//以下操作将该层的每个卷积核都进行了填充，然后进行了DFT变换，用到了FFTW3这个库
	for(size_t i = 0; i < numInputChannels * numOutputChannels; i++)//对于本层网络来说，numInputChannels * numOutputChannels=1*112=112
	{	
		
		pad(tensorRank, ws, kernelDimensions, padded, paddedInputDimensions);//该函数在vetor.cpp中实现，void pad(size_t rank, const nnet_float *input, const size_t *input_dims, nnet_float *output, 
											//const size_t *output_dims);inputDimensions=[28,28]，把kernel的尺寸填充成输入图像的尺寸
		fftwf_execute_dft_r2c(forwardTransform, padded, (fftwf_complex *)fws);//void fftw_execute_dft_r2c(const fftw_plan p,double *in, fftw_complex *out);执行dft变换
											//可以替换的部分，将填充后的kernel进行DFT变换，得到fws
		
		//fft2d_arb_r2c(padded,inputDimensions,(complex<float>*)fws);	//对卷积核进行FFT得到fws，这里先使用基4算法

		//fft2d_dit2_r2c(padded,m_ldn,paddedInputDimensions,(complex<float>*)fws);	//使用基2算法对32x32的图像进行计算,m_ldn = 2,3,4,5..===>4,8,16,32...
		memset(fdws, 0, sizeof(nnet_float) * frequencyVolume);//清零操作
		ws += kernelVolume;//对下一个卷积核进行同样的操作，因为每个卷积核的大小是81或25（kernelVolume）,以下同理
		fws += frequencyVolume;
		fdws += frequencyVolume;
	}
}

void ConvolutionalPadded::endBatch()//在反向传播的过程中，计算误差关于权值的导数的时候，是在频域中计算的，这里，endBatch函数将频域又转到空域中来
{
	nnet_float normaliser = 1.0 / (nnet_float)inputVolume;

	nnet_float *fdws = frequencyDeltaWeights;
	nnet_float *dws = deltaWeights;
	
	for(size_t i = 0; i < numInputChannels * numOutputChannels; i++)
	{
		fftwf_execute_dft_c2r(backwardTransform, (fftwf_complex *)fdws, padded);//执行逆DFT变换，输出在padded中，void fftw_execute_dft_c2r(const fftw_plan p,fftw_complex *in, double *out);
		
		//fft2d_arb_c2r((complex<float>*)fdws,inputDimensions,padded);		
		//fft2d_dit2_c2r((complex<float>*)fdws,m_ldn,paddedInputDimensions,padded);	
		extract_valid_rotate(tensorRank, padded, paddedInputDimensions, dws, kernelDimensions, normaliser);//在vector.cpp中实现，得到kernelDimensions维度的dws，而且是旋转过的，对应于
											//下面的Convolutional_Kernel::calculateGradients，实际上执行的是Notes on Convolutional_Kernel Neural Networks的公式（7）

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
	for(size_t i = 0; i < numInputChannels; i++)//对于l2层，numInputChannels是1
	{	
		
		pad(tensorRank, features, inputDimensions, padded, paddedInputDimensions);//效果相当于直接在features（图像矩阵）的行和列后面补零
		fftwf_execute_dft_r2c(forwardTransform, padded, (fftwf_complex *)ffs);//可以替换的地方，将features进行DFT变换成ffs
		
		//fft2d_arb_r2c(padded,inputDimensions,(complex<float>*)ffs);
		//fft2d_dit2_r2c(padded,m_ldn,paddedInputDimensions,(complex<float>*)ffs);
		
		features += featureVolume;
		//features += inputVolume;//加一个inputVolume就表示跳到下一幅图像（特征）
		ffs += frequencyVolume;//同上，在频域中
	}

	for(size_t i = 0; i < numOutputChannels; i++)//对于l2层，outputChannels是112
	{
		ffs = frequencyInputs;//这里又回到原来的第一幅频域输入图像，尽管下面加了frequencyVolume
		memset(fas, 0, sizeof(nnet_float) * frequencyVolume);//给fas赋初值

		for(size_t j = 0; j < numInputChannels; j++)
		{
			vector_complex_fma(fas, fws, ffs, frequencyVolume / 2);//重要！！！复数相乘，即将变换后的核fws与变换后的输入图像ffs在频域中相乘，结果在fas所指的内存中，frequencyVolume是840，除以2表明有420个这样的复数
			fws += frequencyVolume;
			ffs += frequencyVolume;
		}

		fas[0] += biases[i] * (nnet_float)inputVolume;//更新fas[0]，为什么要这么做？

		fftwf_execute_dft_c2r(backwardTransform, (fftwf_complex *)fas, padded);//可以替换的部分，将频域内得到的激活值经过IDFT变换为空域中的值，结果存在了padded中
		//fft2d_arb_c2r((complex<float> *)fas,inputDimensions,padded);
		//fft2d_dit2_c2r((complex<float> *)fas,m_ldn,paddedInputDimensions,padded);
		extract_valid(tensorRank, padded, paddedInputDimensions, as, outputDimensions);//提取padded矩阵右下方的有效值，于是as就是空域中的有效的卷积之后的激活值

		vector_scale(as, outputVolume, normaliser);	//计算逆变换时需要乘以normaliser

		as += outputVolume;//下一个激活值图像
	}

	(*activationFunction)(activations, deltaActivations, numOutputs);//activationFunction=func,在构造函数中已经被初始化，在空域中计算最终的激活值，顺便初始化该层的deltaActivations
}

void ConvolutionalPadded::backward(nnet_float *bpDeltaErrors)
{
	nnet_float *fdes = frequencyDeltaErrors;
	nnet_float *fws;//频域中的权值
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

		fftwf_execute_dft_c2r(backwardTransform, (fftwf_complex *)ftemp, padded);//需要替换的部分
		//fft2d_arb_c2r((complex<float> *)ftemp,inputDimensions,padded);
		//fft2d_dit2_c2r((complex<float> *)ftemp,m_ldn,paddedInputDimensions,padded);
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
		//complex<float> *temp=new complex<float>[inputDimensions[0]*inputDimensinos[1]];
		deltaBiases[i] += vector_sum(des, outputVolume);//初始的偏置量加上改成输出的结果图像上所有像素点的delta值的和=该层第i个输出通道的偏置

		pad_rotate(tensorRank, des, outputDimensions, padded, paddedInputDimensions);//将des矩阵旋转180度之后在行和列后面补零，由于这里处于向后反馈的过程，因此要把deltaError补成inputDimensions这样大
		fftwf_execute_dft_r2c(forwardTransform, padded, (fftwf_complex *)fdes);//然后再将填充后的deltaError进行DFT变换
		//fft2d_arb_r2c(padded,inputDimensions,(complex<float>*)fdes);
		//fft2d_dit2_r2c(padded,m_ldn,paddedInputDimensions,(complex<float>*)fdes);
		//fdes=reinterpret_cast<fftwf_complex*>temp;

		for(size_t j = 0; j < numInputChannels; j++)
		{
			vector_complex_fma(fdws, fdes, ffs, frequencyVolume / 2);//将空域中的deltaError与输入图像做卷积，这里换成在频域中相乘，得到fdws,注意，这里实际上做的是互相关操作，之前已经将padded旋转过了
			ffs += frequencyVolume;
			fdws += frequencyVolume;
		}

		des += outputVolume;//下一个输出图像
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
