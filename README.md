# Channelformer
Code for Channelformer: Attention based Neural Solution for Wireless Channel Estimation and Effective Online Training

We propose an encoder-decoder structure (called Channelformer) which exploits the attention mechanism to focus on the most important input information. 

Run Demonstration_of_H_Rayleigh_Propogation_Channel and Demonstration_of_H_Doppler_Propogation_Channel for test the trained neural networks on the extended SNR and Doppler shifts. 
Run Dynamic_adaptation_batch for online training test

%% File +Training has 

		ResNN_pilot_regression to train the ReEsNet and InterpolateNet. 
		Training_hybrid_channelformer to train online Channelformer. 
		Training_hybrid_offline to train offline Channelformer. 
		Training_Transformer to train TR structure. 
		Online_training offer a chance to train the neural network with sufficient samples, to adjust these samples with 10 epoch. 

%% File +parameter has 

		parameters contains the system parameters for generating the training data and testing on the extended SNR
		parameters_doppler contains the system parameters for testing on the extended Doppler shift
		parameters_residual_neural_network contains the hyperparameters for the decoder
		parameters_online_average contains the hyperparameters for online training

%% File +Channel contains 

		Rayleigh_fading_channel is a theoretical Rayleigh fading channel
		Propagation_Channel_Model is a LTEfading channel developed by MATLAB specificed in https://uk.mathworks.com/help/lte/ref/ltefadingchannel.html. 
		We use generalized method of exact Doppler spread method for channel modelling. 
		Propagation_Channel_Model_Online is the channel set for the online training 

%% File +CSI has

		LS - It is the implementation of the LS method and the time interpolation method is bilinear method. 
		MMSE - It is the linear MMSE method and the time interpolation method is bilinear method. 
		MMSE_Uniform_PDP - It is the estimate sMMSE method, which is how we generate the MMSE label for online training

%% File +Data_Generation contains

		Data_Generation - used to generate the training data for online Channelformer offline
		Data_Generation_Online - used to generate the training data for online training
		Data_generation_offline_version - used to generate the training data for offline Channelformer and HA02. 
		Data_Generation_Residual - used to generate the training data for InterpolateNet and ReEsNet
		Data_Generation_Transformer - used to generate the training data for TR method. 

%% File +OFDM contains 

		OFDM_Receiver - OFDM receiver
		OFDM_Transmitter - OFDM transmitter
		Pilot_extract - extract the pilot 
		Pilot_Insert - insert the pilot 
		QPSK_Modualtor - generate QPSK symbols 
		QPSK_Demodulator - decode the received QPSK signals

%% File +Pruning has

		Encoder_Pruning - used to prune the encoder of Channelformer, which is part of the code for Hybrid_Pruning for initial test
		Hybrid_Pruning - used to prune the Channelformer
		Fine-tuning - used to fine-tune the pruned neural networks
		Residual_NN_Pruning - used to prune DAG network (trained InterpolateNet and ReEsNet) only 

%% File Residual_NN contains 

		Interpolation_ResNet - Untrained InterpoalteNet (WSA paper)
		Residual_transposed - Untrained ReEsNet
		Residual_resize - Untrained ReEsNet (transpoed convolutional layer by bilinear interpolation)

%% File +transformer_HA02 (same as VTC - +transformer file) contains 

		model_transformer - system model for TR
		model - system model for HA02
			Encoder_block - the encoder of HA02 
			Decoder_block - the decoder of HA02
		+layer contains the layer modules for attanetion mechanism and residual convolutional neural network
			normalization - layer normalization
			FC1 - fully-connected layer
			gelu - Activation function of GeLu
			multiheadAttention - multihead attention module, which calcualte the attention from Q, K and V
			attention - main control unite of the multiohead attention module, designed by tranformer encoder
			FeedforwardNN - feedforward neural network designed by tranformer encoder

%% File +transformer contains Channelformer code 

		model - system model for Channelformer
		+HA03 - The encoder and decoder architecture of Channelformer
			Encoder_block - the encoder of Channelformer 
			Decoder_block - the decoder of Channelformer
		+layer contains the layer modules for attanetion mechanism and residual convolutional neural network
			normalization - layer normalization
			FC1 - fully-connected layer
			gelu - Activation function of GeLu
			multiheadAttention - multihead attention module, which calcualte the attention from Q, K and V
			attention - main control unite of the multiohead attention module, designed by tranformer encoder
			FeedforwardNN - feedforward neural network designed by tranformer encoder

%%% Comments 

Run with MATLAB 2021B, with fully-installed deep learning toolbox because it requires customized training. 

The authors gratefully acknowledge the funding of this research by Huawei. 
