# How to use CORAL

This tutorial will walk through how to run CORAL and visualize results. 

## Input setup

There are two main input types for CORAL: pipelines and scenarios. The pipeline is the csv file with all project specific data.

![input_csv](../images/input_csv.png)

Examples of pipelines can be found in library/pipelines. This is where the user should save their pipeline.

The scenario contains all vessel and port information as well as the associated pipeline for a CORAL run. 

```{code-block} python
description:
- 'Example Altantic pipeline 1'
pipeline: example_atlantic_2024
allocations:
  ahts_vessel: !!python/tuple
  - example_ahts_vessel
  - 2
  feeder: 
  - !!python/tuple
    - example_heavy_feeder_1kit
    - 4
  - !!python/tuple
    - example_feeder
    - 4
  towing_vessel: !!python/tuple
  - example_towing_vessel
  - 10
  wtiv:
  - !!python/tuple
    - example_heavy_lift_vessel
    - 3
  - !!python/tuple
    - example_wtiv
    - 1
  port:
  - !!python/tuple
    - new_london
    - 1
  - !!python/tuple
    - new_bedford
    - 1
future_resources:
- - wtiv
  - example_wtiv
  - - 2030-01-01
remove_resources:
- - wtiv
  - example_heavy_lift_vessel
  - - 2035-01-01
```

## Running CORAL
The below command in the command line will run run_coral.py, running each specified scenario and saving the results in individual csv files within the folder with the folder name provided at postprocessing/results.

```{code-block} python
python run_coral.py foldername scenario1 scenario2 scenario3

```

The line below would be used to run the example scenarios provided.

```{code-block} python
python run_coral.py atlantic_example example_atlantic_1 example_atlantic_2 example_atlantic_3
```

## Running postprocessing script
The below command will run coral_postproc.py which creates a summary slide deck, comparing the runs in the given folder. 
```{code-block} python 
python coral_postproc.py foldername
```
The figures included in the slide deck are created in the coral_plotting.py file. 


