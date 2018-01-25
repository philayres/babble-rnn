# func-26-1-9 Notes

2017-12-11 11:33:28

Run with arguments ./test/critiquepracticalreason_00_kant_64kb.c2cb-3200 

## Description

Attempt to refine the decoder to improve autoencoder loss

At 200 stopped training encoder and decoder and started training generator.
Working with short corpus. At this point the encoder / decoder loss is at 0.68

At 430 stopped, changed the loss weights to 0.95 in favour of the 
generator before the decoder, since that loss had jumped (relative to itself)
during this training run.

At 510 stopped and changed the loss weights to 1.0 in favour of the 
generator to try and force the loss down further.

Return to weights at 200, correcting the generator of the 
next step encoded output. This was incorrectly generating the 
output of the generator, which of course then attempted to learn against
the values it had produced itself. Resolved this by adding a fourth output
which is the output of the encoder, which has learnt based on the 
autoencoder training at the first stage.

At 705 continue, but with generator getting 99% of the loss
weighting, splitting the relative losses more evenly.

At 866 switched to RMSprop. This had an immediate improvement in losses.

At 934 switched generator loss to mean absolute error and 100% weight.

At 1930 tried adding a third LSTM to the generator.

At 2560 added a fourth LSTM to the generator.

At 3150 added 5th LSTM

At 4040 removed the 5th LSTM and swapped in
a wide GRU in parallel with the LSTMs. Use a concatenation
and the fifth LSTM after to keep things in the right
size for the partially trained model
