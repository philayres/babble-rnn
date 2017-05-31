babble-rnn: Generating speech from speech with LSTM networks
==

This code, associated models and Jupyter notebook accompany the *babble-rnn* research project in the use of machine learning to generate new speech by modelling human speech audio, without any intermediate text or word representations. The idea is to learn to speak through imitation, much like a baby might. The goal is to generate a babbling audio output that emulates the speech patterns of the original speaker, ideally incorporating real words into the output.


The implementation is based on Keras / Theano, generating an LSTM RNN. The resulting models have learned the most common audio sequences of a single 'performer', and can generate a probable audio sequence when provided a seed sequence.

The network relies on Codec 2, an open source audio compression and encoding codec that is designed to deliver low bitrate representations of the source, while maintaining a clear representation of the underlying attributes of the audio. 

Read the [babble-rnn tech post](docs/babble-rnn-generating-speech-from-speech-post.html)

View the [babble-rnn code on Github](https://github.com/philayres/babble-rnn/blob/master/README-code.md)


License for babble-rnn
--

Copyright (c) 2017 Phil Ayres https://github.com/philayres

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
