# bootstrap PLS
Files contain methods to produce Design Space (DS) and probabilisitc predictions. 

## Requirements 
All the code was run on python 3.9.9 with the following package versions:

#### Packages 
|Package| Version|
|-------|--------|
|numpy           |          1.22.3|
|matplotlib         |       3.5.1|
|pandas           |         1.4.2|
|scipy              |       1.8.0|

## Structure of the Repository

### PLS.py

The class PLS contained in PLS.py contains all of the methods used to make probabilistic predictions. 

### Recreating Figures from the Paper

The figures from the paper can be produced by running the file with the name corresponding to the name of the example in the paper. 

e.g. Reproducing the figures in the first example of the paper requires running the "Simple_Design_Space_and_Prediction.py" script. 

### Helper files 

"MCMC.py" is used to generate samples for the reductive amination. 



## Contributors 

| Contributor      | Acknowledgements          |
| ---------------- | ------------------------- |
| James Odgers     | The research was funded by an Engineering and Physical Sciences Research Council / Eli Lilly Prosperity Partnership (EPSRC EP/T005556/1) and by Eli Lilly \& Company|
