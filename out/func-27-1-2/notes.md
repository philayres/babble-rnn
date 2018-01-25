# func-27-1-2 Notes

2017-12-14 12:45:41

Run with arguments ./test/critiquepracticalreason_00_kant_64kb.c2cb-3200 none

## Description

Decent autoencoder with limited encoder parameters.

At 440, stopped and added some complexity to the decoder
to bring down the 
loss of the autoencoder stage.

At 960 stopped. Reverted to weights with from 440. 
Added 2 GRU layers in the decoder. This continued in the 27-1-3, since the weights wouldn't load.

