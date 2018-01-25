# func-27-1-22 Notes

2018-01-12 12:49:20

Run with arguments ./test/critiquepracticalreason_00_kant_64kb.c2cb-3200 

## Description

Moving the softmax to cover the very last point in the encoder, before the digested input is concatenated with the result.

Compare with the previous, where at iteration 1000:

generator loss ~0.14
model_2_loss_1 (auto encoder loss) ~2.3
total loss ~0.2

This version

generator loss ~0.027
model 2 loss ~2.8
total loss ~0.09

Looks like a smooth but slow drop in loss. 
Continuing from 15000 to 25000 iterations
