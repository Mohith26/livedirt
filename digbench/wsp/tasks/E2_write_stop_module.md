WRITE (Exp-5/6 style). The target tensors of a NEW small module start at zero. Write weights for a module that takes the
same 7 input channels and outputs, per bite, the fraction of the planned path the bucket completes before the machine
stalls: compute the cut depth under the bucket along the path from channels 0-1, the passive-wedge resistance with the
soil from channels 4-6, accumulate along x in the drag direction, and stop where it exceeds the machine limit. Program
it layer by layer (convolutions, ReLU thresholds, cumulative sums as fixed triangular matrices). You may use the
forward-pass budget to compare your module's completion with the true completion on practice bites and refine it.
Target: stall classification >= 91% and median stop-point error <= 3% of the path on validation.
