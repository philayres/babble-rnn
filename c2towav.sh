fn=$1

#~/store/codec2/Vocoder1300/test/c2dec $fn  $fn.raw charbits
#sox -r 8000 -b 16 -c 1 -e signed  $fn.raw $fn.wav                                                    
~/personal_projects/codec2-babble/build_linux/src/c2dec 3200 $fn $fn-3200.raw --plainbytes
sox -r 8000 -b 16 -c 1 -e signed  $fn-3200.raw $fn-3200.wav
