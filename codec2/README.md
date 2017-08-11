Codec2 Binaries
===

The adapted **Codec 2** binaries have been included here for convenience. Note the license in COPYING.

This version of Codec 2 has been adapted to allow 'plainbytes' output from c2enc and c2dec utilities, when 
running the 3200 bit rate. Other bit rates have not been tested.

**plainbytes** outputs each parameter of a Codec 2 frame as an unsigned char (a single byte), for easy 
reading by babble-rnn.

Full credit for Codec 2 goes to David Rowe, and all rights to the Codec 2 project are unchanged. Full details at: http://www.rowetel.com/?page_id=452

The intention for this small adaptation is to publish the changes separately from Codec 2 so that 
interested developers can incorporate them into the latest version, and to contribute the changes back to the 
main line project. Currently the Codec 2 project is large and controlled in Subversion, so I'm not keen on 
generating a whole project specifically for babble-rnn.