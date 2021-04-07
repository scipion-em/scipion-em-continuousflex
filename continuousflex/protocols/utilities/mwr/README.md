# Missing wedge restoration (MWR)

The code in this repository is described in [this pre-print](https://hal.inria.fr/hal-01966821/document). This paper has been submitted to Journal of Structural Biology and is currently under revision.

## Contents
- [System requirements](##System requirements)
- [Installation guide](##Installation guide)
- [Instructions for use](##Instructions for use)

## System requirements
__MWR__ has been implemented using Matlab R2015a. It has been tested on Linux (Debian 8.6) and Mac OSX (10.12.3) and should also work on Windows.

### Dependencies
Our code needs following conditions to run:
- Matlab Image Processing Toolbox to display images (the processing itself does not depend on this toolbox)
- BM4D denoising algorithm implemented by M. Maggioni

## Installation guide
Download the BM4D implementation [here](http://www.cs.tut.fi/~foi/GCF-BM3D/BM4D_v3p2.zip), and unzip the folder in utils/. That's it.

## Instructions for use
Instructions for using MWR are contained in the Matlab scripts example_*.m. 
- example_proteasome.m applies the method on synthetic data.
- example_gold_particle.m applies the method on experimental data.