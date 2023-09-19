# landscape_sketch_and_step
Supplementary material for the paper 

```"Landscape-Sketch-Step: An AI/ML-Based Metaheuristic for Surrogate Optimization Problems",``` by R. Monteiro K. Sau (2023).


A [preprint is available on Arxiv](http://arxiv.org/abs/2309.07936).

```
@misc{monteiro2023landscapesketchstep,
      title={Landscape-Sketch-Step: An AI/ML-Based Metaheuristic for Surrogate Optimization Problems}, 
      author={Rafael Monteiro and Kartik Sau},
      year={2023},
      eprint={2309.07936},
      archivePrefix={arXiv},
      primaryClass={cs.LG}}
```

Comments:
* All the data is available in the file `LSS_data.zip`.
* The main module and all the available in the folder `LIBS`.
* The script `query.py` plays the role of a connector. It only exists due to the code architecture, initially designed to connect the LSS implementation with Molecular Dynamics solvers.
* The main script is `toy_problem_mlp_with_argument.py`. All the examples given in the paper are run through that piece of code.
