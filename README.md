## multicell-sims
This contains the source code for the poster "Role of Intercellular Interactions on Single Cell and Population Level Responses: Considerations for Multicellular Bioreporter Design" published at RSGDREAM24 at https://iscb.junolive.co/rsgdream/live/exhibitor/rsgdream2024_poster_54

Notebooks and scripts to parse Virtual Cell VCML files and run multicellular simulations on them using the Giellspie algorithm. Implemented in python using `numpy` and `numba` with plots from `matplotlib`

`kinetics2.ipynb` contains the all of the code use to generate the data presented. `SF_detector.vcml` is the XML representation of a Virtual Cell BioModel that is being parsed to create a multicellular simulation environment.
Detailed applications of its usage can be found in the paper by the same name when it is published as conference proceedings in Bioinformatics. 
