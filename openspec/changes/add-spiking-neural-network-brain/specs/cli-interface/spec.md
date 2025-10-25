# CLI Interface Capability - Delta Specification

This document specifies the changes needed for the command-line interface to support spiking neural network brain architecture selection.

## MODIFIED Requirements

### Requirement: Brain Type Argument Extension
The CLI argument parser SHALL accept "spiking" as a valid brain type option.

#### Scenario: Spiking Brain Selection
**Given** a user wants to run simulation with spiking neural network  
**When** they execute `python scripts/run_simulation.py --brain spiking`  
**Then** the CLI SHALL parse "spiking" as a valid brain type  
**And** SHALL pass the selection to the brain factory  
**And** SHALL not raise validation errors  

#### Scenario: Brain Type Help Text
**Given** a user requests help for brain type options  
**When** they execute `python scripts/run_simulation.py --help`  
**Then** the help text SHALL list "spiking" among valid brain types  
**And** SHALL provide brief description of spiking neural network approach  

### Requirement: Configuration Compatibility
The CLI SHALL support loading spiking brain configurations through existing configuration mechanisms.

#### Scenario: Spiking Configuration Loading
**Given** a YAML configuration file with spiking brain parameters  
**When** loaded via `--config spiking_medium.yml`  
**Then** the CLI SHALL parse the configuration  
**And** SHALL validate spiking-specific parameters  
**And** SHALL initialize the spiking brain with specified parameters  

#### Scenario: Brain Type Override
**Given** a configuration file specifies a different brain type  
**When** user provides `--brain spiking` CLI argument  
**Then** the CLI argument SHALL override the configuration file  
**And** SHALL use spiking brain regardless of config file brain type  

### Requirement: Error Handling
The CLI SHALL provide clear error messages for spiking brain configuration issues.

#### Scenario: Invalid Spiking Parameters
**Given** a configuration with invalid spiking brain parameters  
**When** the CLI attempts to initialize the brain  
**Then** SHALL provide specific error message about invalid parameters  
**And** SHALL suggest valid parameter ranges  
**And** SHALL exit gracefully with appropriate error code  

#### Scenario: Missing Dependencies
**Given** spiking brain is selected but required dependencies are missing  
**When** the CLI attempts initialization  
**Then** SHALL provide clear error about missing dependencies  
**And** SHALL suggest installation commands
