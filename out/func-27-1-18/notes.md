# func-27-1-18 Notes

2018-01-07 21:17:53

Run with arguments ./test/critiquepracticalreason_00_kant_64kb.c2cb-3200 

## Description

Removed parallel dense layer in generator. Increased the
size of the LSTMs considerably.

Restart using SGD at 1100 after blowout.

Increased learning rate at 2254

Restart at 1100 again with lr = 0.001

Restart at 6317 with new loss weights to favour generator
