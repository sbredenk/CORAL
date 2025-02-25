#  CHANGELOG

## 0.1 (24 September 2024)

### Model Capability Development
- Combine fixed and floating modules so pipelines can run with both
- Combine monopile, jacket, gravity-based foundation, and semisubmersible foundation to the same module for running within one pipeline
- Add US WTIV vessel type, tracked separately from other WTIV, different feeder assumptions for both
- Add separate tracking for heavy feeders and feeder barges
- Expand port library
- Provide Atlantic examples (pipeline and scenarios)
- Add installation phase completion times to output df for vessel utilization prost processing
- Reorganize file path to move analysis directories to library

### Run Script and Post Processing
- run script in highest directory for command line run command
- 'postprocessing' folder with post processing script (also command line run command) as well as 'coral_plotting.py' with detailed plotting routines 
- Plotting capabilities include vessel utilization, port throughput, cumulative installed capacity, and gantt charts

### Documentation
- Basic README for installing the model and dependancies

## 1.0 (1 March 2025)

### Model Capability Development
- Added ability to remove vessels from pool
- Updated examples
- Made phase overlap an easily accessible variable (in scenario yaml)
- Added ability to specify a port downtime between projects (in scenario yaml)
- Added mooring to floating project installation
- Ability to add specific routed distance from ports to project