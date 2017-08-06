#!/bin/bash
for word in $*; do 
fn=${word/.mp3/}
lame $fn.mp3 -m m --decode $fn.wav
sox $fn.wav -c 1 -r 8000 -b 16 -e signed $fn-8k.wav
sox $fn-8k.wav -c 1 -r 8000 -b 16 -e signed $fn-8k.raw

#path_to_c2=~/personal_projects/codec2-babble/build_linux/src
path_to_c2=./codec2
#~/store/codec2/Vocoder1300/test/c2enc $fn-8k.raw $fn.c2cb charbits  
LD_LIBRARY_PATH=$path_to_c2 $path_to_c2/c2enc 3200 $fn-8k.raw $fn.c2cb-3200 --plainbytes

#~/store/codec2/Vocoder1300/test/c2dec $fn.c2cb $fn.c2cb.raw charbits  
LD_LIBRARY_PATH=$path_to_c2 $path_to_c2/c2dec 3200 $fn.c2cb-3200 $fn.c2cb-3200.raw --plainbytes

sox  -c 1 -r 8000 -b 16 -e signed $fn.c2cb-3200.raw $fn.c2cb-3200.wav

cp $fn.c2cb ~/store/c2gen/test

done
 