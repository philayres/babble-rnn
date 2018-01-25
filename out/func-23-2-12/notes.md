# func-23-2-12 Notes

2017-12-06 17:02:50

Run with arguments ./test/criticalpracticalreason.c2-3200 

## Description

Reduce fit_batch_size, since with timedistributed learning we are getting very few updates.
Maybe this is why learning has stalled.
Dropped to 128 LSTM size in the generator. There is some indication in discussions
that binary scaled items may perform better.
