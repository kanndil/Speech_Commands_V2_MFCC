# Fixed MFCC Generator

## Overview

This project generates fixed Mel-Frequency Cepstral Coefficients (MFCC) from audio files. It includes a simple application that reads WAV files and computes their MFCCs using the provided implementation.

## Files

- `dr_wav.h`: Header file for reading WAV files.
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
   make
   ```

   This will create the executable `fixed_mfcc_generator` in the `bin` directory.

## Usage

After building the project, you can run the executable to process WAV files. Currently, no command-line arguments are implemented, so you will need to modify `main.cpp` to set the input WAV file and parameters.

### Example

Edit `main.cpp` to specify your WAV file and any necessary parameters for the MFCC computation.

## Cleaning Up

To remove compiled files and directories, use:

```bash
make clean
```

This will delete the `obj` and `bin` directories.

## Dependencies

- `g++` for compiling the C++ code.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
