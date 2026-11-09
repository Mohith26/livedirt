READ (calibration, no edits). Answer from the weights (you may analyse them in Python, and use `feature_maps` but not
the error probes):
1. Which input channel has the largest effect on the scalar head's completion output, and with which sign?
2. Does the network use the soil channels (4-6) mostly in the encoder or the decoder? Name the 3 e1 filters most tied to them.
3. Which e4 channels most plausibly encode "the bucket will stall"? Rank 5.
4. In d1/out, which channels write mainly where the bucket cut (channel 2 = 0) vs outside it?
Give confidence for each. These answers are scored against activation-patching measurements you cannot see.
