fn=$1
path_to_c2=~/personal_projects/codec2-babble/build_linux/src
#path_to_c2=~/store/codec2/Vocoder1300/test

#$path_to_c2/c2dec $fn  $fn.raw charbits
#sox -r 8000 -b 16 -c 1 -e signed  $fn.raw $fn.wav                                                    
LD_LIBRARY_PATH=$path_to_c2 $path_to_c2/c2dec 3200 $fn $fn-3200.raw --plainbytes
sox -r 8000 -b 16 -c 1 -e signed  $fn-3200.raw $fn-3200.wav
