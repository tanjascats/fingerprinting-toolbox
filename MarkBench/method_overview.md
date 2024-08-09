# Method overview
We include some high-impact fingerprinting and watermarking methods for tabular data in our benchmarking tool. 
Note that these are our own implementations due to the lack of open-source solutions for the respective methods from their original authors.
This at the same time poses as a problem that we want to address with the MarkBench tool and facilitate open-source development in the domain of watermarking and fingerprinting tabular data.

The list below outlines the methods and open-source solutions if available and types of robustness and utility analysis reported.

| Method ID | Authors | Year | Open-source | Robustness | Utility | Dataset(s) |
| :-------- | :-----  | :--- | :---------- | :--------- | :------ | :--------- |
| [AHK (pioneer)](https://courses.cs.washington.edu/courses/cse590q/03au/watermarking_vldbj.pdf) | Agrawal et al. | 2002 | - | Horizontal subset, Vertical subset, Bit-flipping (deterministic, randomised), Mix-and-match, Additive | Mean, Var | [Forest CoverType](kdd.ics.uci.edu/databases/covertype/covertype.html) (int) |
| [**LSJ (pioneer)**](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=2070&context=sis_research) | Li et al. | 2005 | - | Misdiagnosis fh, Bit-flipping, Horizontal subset, Superset attack, Invertibility, Collusion | Mean, Var | [Forest CoverType](kdd.ics.uci.edu/databases/covertype/covertype.html) (int) |
| [**Block**](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=1562&context=sis_research) | Liu et al. | 2005 | - | Bit-flipping, Horizontal subset, _Vertical subset_, _Collusion_, _Additive_ | - | 

<span style="font-size:0.5em;">_Bolded are the fingerprinting methods._</span>

<span style="font-size:0.5em;">_Italic are the attacks dicussed but not analysed._</span>


Challenges (per method):
- AHK - marks only numerical attributes, PK dependent, skips null values
- LSJ - marks only numerical attributes, PK dependent, attribute order-dependent
- Block - not collusion resistant, PK dependent, attribute order-dependent

Extensions (per method):
- AHK
    - varying the marking rates for attributes, user-specified (i.e. the attribute weights in marking)
    - varying the number of candidate bit positions (epsilon assigned to each attribute)
    - VPK such that (i) (for single numerical attribute table) use the MSBs as VPK, the rest for marking, (in case of multiple numerical values) (ii) use the most diverse one as a VPK, or (iii) concatenate MSBs from multiple numerical attributes
- LSJ
    - VPK in Li et al. "Constructing a virtual primary key for fingerprinting relational data." -- use the most significant bits of numerical attributes whose hashes are closest to zero
    - adapted BoSh codes for collusion ("c-secure with eps-error", i.e. it enables the capture of a member of a coalition of at most c members with probability at least 1-eps; i.e. eps is an error)
    - the possibility of adapting Guth Pfitzmann codes for collusion and flipping detection, from Guth & Pfitzmann "Error and collusion-secure fingerprinting for digital data"
    - tracing shortened and corrupted fingerprints from Safavi-Naini & Wang "Traitor tracing for shortened and corrupted fingerprints"