babble-rnn: Generating speech from speech with LSTM networks
==

*babble-rnn* is a research project in the use of machine learning to generate new speech by modelling human speech audio, without any intermediate text or word representations. The idea is to learn to speak through imitation, much like a baby might. The goal is to generate a babbling audio output that emulates the speech patterns of the original speaker, ideally incorporating real words into the output.

The implementation is based on [Keras](https://keras.io) / [Theano](http://www.deeplearning.net/software/theano/), generating an LSTM RNN; and [Codec 2](http://www.rowetel.com/?page_id=452), an open source speech audio compression algorithm. The resulting models have learned the most common audio sequences of a 'performer', and can generate a probable babbling audio sequence when provided a seed sequence.


Read the [babble-rnn tech post](http://babble-rnn.consected.com/docs/babble-rnn-generating-speech-from-speech-post.html)

View the [babble-rnn code on Github](https://github.com/philayres/babble-rnn/blob/master/README-code.md)

**Wondering what babble-rnn can do?** Listen to the latest babble produced by the experiments since the original tech report:

[play the .mp3](https://github.com/philayres/babble-rnn/blob/v3/out/func-28-1-3/27830-example.mp3)

This babbler is a stack of 11 bidirectional LSTMs, attempting to learning an encoded sequence of data (frame of 13 normalized parameters, representing 20ms of audio).
Groups of LSTMs are trained to, while keeping others locked, to limit the complexity of learning such a deep network.

The audio itself is highly compressed through the Codec 2 (see the original tech post for details) producing a 3200 bit per second
stream of frequency, energy, sinusoidal and voicing parameters. An autoencoder learns the features of this against a particular human
speaker, to compress the output further. The encoder stage is a mix of 2D convolutional layers, picking features from the Codec 2 data,
in parallel with a series of standard hidden layers, before being merged into a single encoded output at a quarter of the rate of the
original Codec 2 input (80ms audio per frame, although more parameters output than the original).
