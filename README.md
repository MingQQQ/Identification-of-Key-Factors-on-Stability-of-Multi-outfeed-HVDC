# The code for the integration of stability indexes and global sensitivity analysis of fusion index in Multi-outfeed HVDC

![Graphical abstract]( /ProjectImages/GraphicalAbstract.png)

Resources and extra documentation for the manuscript "A Global Sensitivity-based Identification of Key Factors on Stability of Power Grid with Multi-outfeed HVDC" published in IEEE Latin America Transactions. The project hierarchy and folders description is as follows

1. **MATLAB_EWF_AHP**: The effectiveness verification and fusion of stability indexes.
   1. `make_plots.m`. The actual power grid simulation data in the engineering software PSD-BPA is used to verify the effectiveness of various stability indexes.
   2. `EWF_AHP.m`. The code for the entropy weight-fuzzy analytic hierarchy process (EWF-AHP) method, after calculating the comprehensive weights, combines multiple indexes in each sample into a single comprehensive index.
2. **PYTHON_GSA**: Global sensitivity analysis of key operational variables in the system that affect the comprehensive index.
   1. `gaussian_process_emulator.m`. The actual power grid sim
   2. `example_gaussian_process_emulator.m`. The code for the entmprehensive index.
   3. `sensitivity_analysis_sobol.m`. The actual
   4. `example_sobol.m`. The code for the entmprehensive index.
