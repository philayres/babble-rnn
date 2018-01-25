# func-26-1-8 Notes

2017-12-10 19:35:21

Run with arguments ./test/critiquepracticalreason_00_kant_64kb.c2cb-3200 --load-weights=out/func-26-1-2/weights-1200.h5

## Description

Attempt to predict the mid layer to use it for later loss calculations
At iteration 444 changed the loss weights to favour the generator 
loss more highly. Also, switched to the long corpus.
At 450 switched the loss weights again to ensure the encoder is adequately
addressed, since it is core to all the other components.
Back to the short corpus after a night of training with no real progress

At 1482 added in the decoder lstm to try and reduce autoencoder loss below 0.78
