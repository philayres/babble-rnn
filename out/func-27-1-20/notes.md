# func-27-1-20 Notes

2018-01-09 18:52:53

Run with arguments ./test/critiquepracticalreason_00_kant_64kb.c2cb-3200 

## Description

Back to LSTM 128 units. Using he_normal initializers on
the generator layers, per the good experience of:

https://obilaniu6266h16.wordpress.com/2016/04/12/keras-he-adam-breakthrough/


Progressed well. But exploded sometime after 3000. Jump back here and restart with RMSprop

At 13288 stopped and set the weight to exclusively look at the generator

Restarted at 13200 with:

    encoder_trainable = False
    decoder_trainable = False
    generator_trainable = True

Just loading the losses to be all for the generator actually
doesn't help since it allows the autoencoder to be made 
into junk to increase generation of simple encoded values.

Jumped back to 13700 at 16778. Reduced learning rate. Trying to see if we can capture the interesting loss drop that happened around this time.

At 16804 stopped again. The autoencoder loss has jumped strangely, despite being set not to be trainable (bug maybe? missed a layer?) Making it trainable with a low loss weighting

Learning has bottomed out. It was a good result overall, but there are definitely opportunities to improve.
