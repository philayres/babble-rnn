# func-21-7-1 Notes

2017-11-30 16:46:55

Run with arguments ./test/criticalpracticalreason.c2-3200 none

## Description

Simplify the input layers to a concatenated set of the main input, allowing Conv2D to see the relationship between parameters that would otherwise have been edges.

Changed the influence of loss from mid output to 0 and moved it to after the Conv layers
