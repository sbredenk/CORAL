# User Guide

This page provides an overview of what CORAL is currently able to model. These capabilities are all demonstrated through example notebooks in the examples folder of the CORAL repository. 

## Base configurations
The base configurations creates a starting point for an ORBIT configuration that CORAL will then update for each project. There is one base configuration for fixed projects and one for floating projects. CORAL will pull the corresponding configuration file depending on the assigned substructure of the project. The configuration files can be found in library/configs.

## Input CSV format
The input csv file contains all the project specific information that changes across projects. Each row corresponds to a project in the pipeline and each column corresponds to an attribute (distance to shore, capacity, turbine rating, etc.) Here is an example of a few rows of the csv:

![input_csv](../images/input_csv.png)

A full example of a CORAL pipeline can be found at library/pipelines.

The sections below detail the vessel configuration choices for the foundation and turbine installation phases. 

The associated port in the csv file above contains two pieces of information: the coordinates of the port and the number of cranes at the port. For the purposes of CORAL, only the coordinates are used. The straight line distance between the coordinates of the project as defined in the input csv and the coordinates of the port is assigned as the distance vessels must travel for installation. Additionally, a routed distance can be specified in the input csv if a routed distance is known. The number of berths at a given port are defined in the allocations. 

## Fixed bottom speifics

### Monopile and jacket substructures
Monopile and jacket foundations have very similar installation structures from a CORAL perspective. Both foundation types require the same classes of vessels, and perform foundation and turbine phases separately. Depending on the substructure indicated in the input csv, the corresponding ORBIT design and installation phases are added to the configuration. For example, a monopile project adds the MonopileDesign and MonopileInstallation ORBIT modules to the project configuration.

```{note}
Foundation and turbine installation phases are overlapped by 80%, meaning turbine installation begins when foundation installation is 20% complete.
```

#### Foundation installation
The foundation installation phase includes floating foundation installation vessels (FFIVs) and feeder barges. Not all vessels are used on every project, as the use of feeders is dependent on whether feeders are enforced or not. To read more about enforcing feeders look to {ref}`label:enforce_feeders`.

#### Turbine installation
By default, the turbine installation phase assigns both a WTIV and 2 heavy feeders to the project. There is an optional configuration for US-flagged WTIVs described here: {ref}`label:us_flag`.

#### Shuttling vs feedering
While "enforce feeders" will instruct every project to feeder, certain other designations will cause projects to feeder rather than shuttle. The ports of New Bedford, Tradepoint Atlantic, and South Brooklyn Marine Terminal cannot accomodate a WTIV or FFIV, so projects assigned to these ports will be forced to feeder for both foundation and turbine phases. Additionally, due to Jones Act specifications, foreign-flagged WTIVs cannot shuttle between US ports and US offshore wind projects so any project not assigned a US-flagged WTIV will use feedering for the turbine installation phase. 

### Gravity-based substructures
Gravity-based foundations have the turbine assembled at the port so there is only one installation phase (the GravityBasedInstallation phase in ORBIT). This phase uses tug boats to tow the assembled foundation and turbine out to the offshore wind site. 

## Floating specifics
Semisubmersible foundation installation is very similar to gravity-based foundation installation from a CORAL perspective. The installation is also one phase (MooredSubInstallation in ORBIT). The semisubmersible installation requires anchor handling tug supply vessels to transport the foundations to the project site. 


## Optional preferences

### Weather profile
Weather profiles can be added to library/weather and are used to dictate weather delays. There are a couple weather profiles as examples. These are the same weather files used by ORBIT.

### Regional ports
The regional ports option can be used if the user does not want to specify ports for each project. 

(label:us_flag)=
### US-flagged WTIVs
Due to Jones Act regulations, foreign-flagged vessels cannot operate between two US ports. Once the foundations have been installed at a project site, that site is considered a port, therefore foreign-flagged WTIVs cannot shuttle between US ports and the project site. Because of this, all non-specific WTIVs assigned in CORAL are required to feeder from all US ports. CORAL allows the user to specify a fleet of US-flagged WTIVs that can shuttle to any US port that can accommodate a WTIV. This can be done by adding a column to the input csv called "us_wtiv" that contains whether the project uses a US WTIV as a boolean. Additionally, the user would need to add the example_wtiv_us to the scenario description.

(label:enforce_feeders)=
### Enforce Feeders
