# The BECCS-Sweden repository
Contains models and data for performing Robust Decision Making [1] analyzes of bioenergy carbon capture in Sweden. 
Please refer to the ModelAppendix.ipynb for a tutorial style introduction to the models.
The repo relies extensively on the [EMA Workbench](https://github.com/quaquel/EMAworkbench) [2].

# Installing and running the model
First make a clone of the BECCS-Sweden repository. Then install EMA Workbench in your Python environment (we use Anaconda, Python v.3.11.5).

    pip install -U ema_workbench

To run the models, navigate to your BECCS-Sweden directory and run e.g. chp_controller.py in the terminal. If the model does not run, check other dependencies. You can also check the [EMA Workbench](https://github.com/quaquel/EMAworkbench) for more installation information. In the controller files, sample sizes can be lowered (e.g. to 1000) for fast model evaluations.

[1] Lempert, R.J. (2019). Robust Decision Making (RDM). Springer eBooks, [online] pp.23–51. doi:https://doi.org/10.1007/978-3-030-05252-2_2.

[2] Kwakkel, J. (2017). The Exploratory Modeling Workbench: An open source toolkit for exploratory modeling, scenario discovery, and (multi-objective) robust decision making. Environmental Modelling & Software, 96, 239–250. https://doi.org/10.1016/j.envsoft.2017.06.054
