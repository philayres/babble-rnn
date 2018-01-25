# func-27-1-8 Notes

2017-12-15 17:59:30

Run with arguments ./test/critiquepracticalreason_00_kant_64kb.c2cb-3200 

## Description

Similar to previous. Added a digested input 
passthrough on the encoder, responding to the 
observation that generated sequences seem to lack
the same volume (energy param) and pitch (Wo param)
range as the seed. Possibly the Convolutional layers
are picking up some good features, but are unable to 
aid in the representation of these levels (rather than
edges and short term trends).

Restart at 2000 with SGD after blowout
