# xEHM

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Evolutionary History Matching

This is a Python package which implements *History Matching* - a statistical analysis technique that calibrates models to data whilst accounting for uncertainty. Currently this is in development and only available as a source installation.

If you would like to test this out, please clone and use the `srcinstall.bat` script to set everything up. Linux/Bash users will need to adjust this script as it was written for the Windows CMD / PowerShell terminals. Please contact cf502@exeter.ac.uk if you are cloning / forking this project as things are changing day-by-day until there is an official release - **it is extremely likely that you will be running old code**. At the moment, this repo is collecting snippets of older code that have passed various tests, the full API is being stress-tested before it goes anywhere near this.

A list of features that are in the process of being uploaded slowly is as follows:

 - n-Dimensional generic / default History matching (I.E. run as a black-box, will do something 'acceptable' with the right inputs)
 - Updated Gaussian Process plugins to support `mogp` as well as `gpflow`
 - A more robust diagnostic / constraint API to critique emulation
 - NROY plotting engine
 - Accelerated low dimensional cases where Rejection Sampling can be eliminated for something much better
 - Evolutionary ladder methods
 - Full save/load/pause/restart controls to facilitate long-running / shared analyses
