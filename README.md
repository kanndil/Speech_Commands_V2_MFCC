# Speech Commands V2 MFCC

## Overview

This project implements an optimized MFCC (Mel-Frequency Cepstral Coefficients) algorithm designed to enhance execution speed on edge devices and reduce memory usage through static memory allocations. MFCC is widely used in audio processing, particularly in Keyword Spotting (KWS) applications, but this implementation can also be experimented with for other uses.

The repository includes a C++ generator that processes the Google Speech Commands dataset to generate a new dataset with the applied MFCC.

### Rationale

The primary motivation behind this repository is to use the optimized MFCC-applied Speech Commands dataset for model training. During testing, we observed information loss when training the model with standard MFCC and performing inference with this optimized MFCC implementation. Training using the optimized dataset should theoretically mitigate this information loss.

## Install Google Speech Commands V2

Follow these steps to install the dataset:

1. Download the Google Speech Commands V2 dataset.
2. Place the dataset in the `/fixed_mfcc_generator/dat` directory.

## Expected Output

After running the generator, the modified dataset will be found in the `/fixed_mfcc_generator/speech_commands_V2_mfcc_version/dat` directory.

## Files

- `main.cpp`: Main entry point for the application.
- `mfcc.cpp`: Implementation of the MFCC generation.
- `mfcc.h`: Header file for the MFCC class.

## Building the Project

To build the project, you need to have `g++` installed. Use the provided `Makefile` to compile the code:

1. **Clone or download the repository.**
2. **Navigate to the project directory:**

   ```bash
   cd fixed_mfcc_generator
   ```

3. **Run `make` to compile the project:**

   ```bash
   make clean
   make
   ```

   This will create the executable `fixed_mfcc_generator` in the `bin` directory.

