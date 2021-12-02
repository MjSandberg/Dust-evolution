This is a module designed to simulate dust-dust interactions in a protoplanetary disk. It is designed as a prototype for the real implementation into a larger code framework called Dispatch.

It is a integrator that solves the Smoluchowski equation with an added momentum conservation component.

One can then input dust sizes and the binning and then evolve in time. From this we get a different distribution of sizes at the end of the run.


To run input dt around 10 years and t_end around 1000 years, with a resolution in the bins of 40. The size range is input in logspace of cm, meaning that -4 would be micron size and 2 would be 100 cm. Using this example runtime should be relatively short.
