# CORAL - Concurrent ORBIT for shared Resource Analysis Library

## Overview
The CORAL tool is an extension of [ORBIT](https://github.com/WISDEM/ORBIT), designed to model the dependence of offshore wind pipelines on shared resources. The main shared resources currently modeled in CORAL are installation vessels and marshaling ports. CORAL takes a csv file, where each row contains the project specificitions needed to run the project in ORBIT, and runs all the projects in order based on the start date with a limited pool of ports and vessels. If a project is set to begin but not all the resources are available, the project will be delayed until the resources are released from other projects. The result is a DataFrame with the start and end time of each project as well as some other attributes that are helpful for plotting purposes. Through the post processing functions included in the CORAL repository, users can analyze deployment rates, annual port throughput, average vessel utilization rates and more. 

CORAL was used to produce results for NREL's [Supply Chain Road Map for Offshore Wind Energy](https://www.nrel.gov/docs/fy23osti/84710.pdf) in the United States. Figure 9 of the report (below) shows the project installation times and project delays for the foundation and turbine installation campaigns of a fixed bottom offshore wind pipeline. 

![roadmap](../images/roadmap-fig.png)

For any questions, please open an issue in the repository or email:
sophie.bredenkamp@nrel.gov.

## Latest Changes


## License

