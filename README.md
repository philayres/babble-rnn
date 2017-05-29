babble-rnn: Generating speech from speech with LSTM networks
==

*babble-rnn* is a research project in the use of machine learning to generate new speech by modelling human speech audio, without any intermediate text or word representations. The aim being to generate a babbling audio output that emulates the speech patterns of the original speaker, ideally incorporating real words into the output.

The implementation is based on [Keras](https://keras.io) / [Theano](http://www.deeplearning.net/software/theano/), generating an LSTM RNN; and [Codec 2](http://www.rowetel.com/?page_id=452), an open source speech audio compression algorithm. The resulting models have learned the most common audio sequences of a 'performer', and can generate a probable babbling audio sequence when provided a seed sequence.


Read the [babble-rnn research post](docs/babble-rnn-generating-speech-from-speech-post.html)

View the [babble-rnn code on Github](https://github.com/philayres/babble-rnn/blob/master/README-code.md)

