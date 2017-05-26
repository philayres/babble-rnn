babble-rnn: Generating speech from speech with LSTM networks
==

This code, associated models and Jupyter notebook accompany the post of the same name describing a research project in the use of machine learning to generate new speech by modelling human speech audio, without any intermediate text or word representations. The aim being to generate a babbling audio output that emulates the speech patterns of the original speaker, ideally incorporating real words into the output.

The implementation is based on Keras / Theano, generating an LSTM RNN. The resulting models have learned the most common audio sequences of a single 'performer', and can generate a probable audio sequence when provided a seed sequence.

The network relies on Codec 2, an open source audio compression and encoding codec that is designed to deliver low bitrate representations of the source, while maintaining a clear representation of the underlying attributes of the audio. 

 