# Method overview
We include some high-impact fingerprinting and watermarking methods for tabular data in our benchmarking tool. 
Note that these are our own implementations due to the lack of open-source solutions for the respective methods from their original authors.
This at the same time poses as a problem we want to address with the MarkBench tool and facilitate open-spource development in the domain of watermarking and fingerprinting tabular data.

The list below outlines the methods and open-source solution if available and types of robustness and utility analysis reported.

| Method ID | Authors | Year | Open-source | Robustness | Utility | Dataset(s) |
| :-------- | :-----  | :--- | :---------- | :--------- | :------ | :--------- |
| [AHK (pioneer)](https://courses.cs.washington.edu/courses/cse590q/03au/watermarking_vldbj.pdf) | Agrawal et al. | 2002 | - | Horizontal subset, Vertical subset, Bit-flipping (deterministic, randomised), Mix-and-match, Additive | Mean, Var | [Forest CoverType](kdd.ics.uci.edu/databases/covertype/covertype.html) (int) |
| More text    | Even more text   | And even more to the right |  |

Challenges:
- AHK - marks only numerical attributes, PK dependent, skips null values

Extensions:
- AHK - varying the marking rates for attributes, user-specified (i.e. the attribute weights in marking)
      - varying the number of candidate bit positions (epsilon assigned to each attribute)
      - VPK such that (i) (for single numerical attribute table) use the MSBs as VPK, the rest for marking, (in case of multiple numerical values) (ii) use the most diverse one as a VPK, or (iii) concatenate MSBs from multiple numerical attirbutes
