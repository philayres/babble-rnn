#!/bin/bash
for word in $*; do 
fn=${word/.mp3/}
lame $fn.mp3 -m m --decode $fn.wav
sox $fn.wav -c 1 -r 8000 -b 16 -e signed $fn-8k.wav
sox $fn-8k.wav -c 1 -r 8000 -b 16 -e signed $fn-8k.raw

~/store/codec2/Vocoder1300/test/c2enc $fn-8k.raw $fn.c2cb charbits  

~/store/codec2/Vocoder1300/test/c2dec $fn.c2cb $fn.c2cb.raw charbits  

sox  -c 1 -r 8000 -b 16 -e signed $fn.c2cb.raw $fn.c2cb.wav

cp $fn.c2cb ~/store/c2gen/test

done
 