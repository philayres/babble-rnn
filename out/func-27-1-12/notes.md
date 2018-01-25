# func-27-1-12 Notes

2017-12-18 12:58:07

Run with arguments ./test/critiquepracticalreason_00_kant_64kb.c2cb-3200 none

## Description

Same as previous one, but training with new weights from start

Continue at 224 with weighting giving balance to generator output,
due to its relatively small parameter values leading to small calculated losses.

Continue at 774 with loss weighting bring in just a little
of the main output calculation, since this 
is perhaps more likely to allow closer matching to 
actual pitch and volume characteristics.

That said, the 3 pass through elements between encoder
decoder pair should have helped this happen, so the concern
here is that they are not actually providing information to 
assist with truly moving the simulated audio in a natural direction.
Perhaps they are too small relatively to manage this and
require scaling, or maybe they are actually modelling something unexpected.


At 1697 changed the weights to allow more weighting to main loss

At 6542 changed the weights to put even more focus on the final output
