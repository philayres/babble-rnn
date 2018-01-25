# func-23-1-7 Notes

2017-12-05 21:04:39

Run with arguments ./test/criticalpracticalreason.c2-3200 

## Description

Share the decoder layers as a model both to learn the current frames, and to 
act as an encoder after the generator runs on encoded data.

Had to remove the named outputs and rely on the ordering of a list.
