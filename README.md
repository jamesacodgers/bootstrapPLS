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

The figures from the paper can be produced by running the relevant file within the "Examples" folder. The file with the name corresponding to the name of the section title of the paper will recreate the relevant figures. 

e.g. Reproducing the figures in the first example of the paper requires running the "Simple_Design_Space_and_Prediction.py" script. 




## Contributors 

| Contributor      | Acknowledgements          |
| ---------------- | ------------------------- |
| James Odgers     | The research was funded by an Engineering and Physical Sciences Research Council / Eli Lilly Prosperity Partnership (EPSRC EP/T005556/1) and by Eli Lilly \& Company|


### Data Sources
Data for the High Shear Wet Granulation example was produced by Vemavarapu _et al._ [1]. The data for the spectroscopy example was produced by Hetrick _et al._ [2]. 

Code for producing the data used for the Michael Addition Reaction was provided by Kennedy Kusumo (https://github.com/KennedyPutraKusumo). Details on the mathematical implementation can be found in [3].

## Bibliography 
[1] Vemavarapu, C. and Badawy, S.I., 2019. Role of Drug Substance Material Properties in the Processibility and Performance of Wet Granulated Products. In Handbook of Pharmaceutical Wet Granulation (pp. 387-419). Academic Press.

[2] Hetrick, E.M., Shi, Z., Barnes, L.E., Garrett, A.W., Rupard, R.G., Kramer, T.T., Cooper, T.M., Myers, D.P. and Castle, B.C., 2017. Development of near infrared spectroscopy-based process monitoring methodology for pharmaceutical continuous manufacturing using an offline calibration approach. Analytical chemistry, 89(17), pp.9175-9183.

[3] Kusumo, K.P., Gomoescu, L., Paulen, R., García Muñoz, S., Pantelides, C.C., Shah, N. and Chachuat, B., 2019. Bayesian approach to probabilistic design space characterization: A nested sampling strategy. Industrial & Engineering Chemistry Research, 59(6), pp.2396-2408.