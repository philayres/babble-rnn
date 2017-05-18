fn=$1

~/store/codec2/Vocoder1300/test/c2dec $fn  $fn.raw charbits
sox -r 8000 -b 16 -c 1 -e signed  $fn.raw $fn.wav                                                    
