# func-28-1-3 Notes

2018-01-29 15:11:06

Run with arguments test/critiquepracticalreason_00_kant_64kb.c2cb-3200 --load-weights=out/func-28-1-2/weights-15000.h5

## Description

Added in two new LSTMs and reloaded weights from last run.

Disabled training on the 3 initial LSTMs and on encoder and decoder

Going well!

Restarted at 3000, having reenabled trainable flag for 
the initial LSTMs, allowing all LSTMs to refine themselves.

Restarted at 5450. Added 2 more LSTMs on the end, disabling the training 
of existing early and mid LSTMs.

