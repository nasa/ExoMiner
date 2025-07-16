# Common issues & Troubleshooting

We list here some of the common issues a user can run into when trying to use the Podman image for the ExoMiner 
Pipeline.

## Pulling the image for a different system architecture

If you pull an image built for an architecture that does not match your system's architecture, you will run into 
compatibility issues when attempting to run that container. Make sure that you pull the image built for your system's 
architecture.

## Permission issues

You attempt to run [run_podman_application.sh](/exominer_pipeline/run_podman_application.sh), but you cannot run it due 
to not having executable permission to run the file. Fix that permission.

## Errors while running the pipeline

Here, we list the set of most common errors that may happen when running the pipeline.

- No stable connection: without it, you cannot access the MAST to download the light curve FITS files and DV XML data, 
and the Gaia servers to download the RUWE for the queried TIC IDs. See if there are any data products in the 
[manifest file](/docs/running-exominer-pipeline.md#outputs) that were not downloaded successfully.
- Incorrect input format: the TIC IDs and sector runs are provided in a way that does not follow the expected format. 
We detail the structure of the inputs in [running-exominer-pipeline.md](/docs/running-exominer-pipeline.md).
- Missing data or non-existing TIC ID: the data are not available for a given TIC ID and sector run in the MAST. It 
might happen that the data are still not available at the MAST, the TESS SPOC pipeline was not run for that case, or 
the TIC ID was not observed for that sector run and/or data collection mode (i.e., 2-min or FFI data). One way to check 
that is to see whether the expected target light curve FITS files and DV XML files were downloaded. If the naming 
convention of the files changes, that can also lead to errors in the pipeline since some aspects rely on the filenames 
having a specific structure.

## I am stuck - What now?

If none of the points [above](#errors-while-running-the-pipeline) resolved your issue, then a good starting point is to understand at which point the 
pipeline failed. Check the results produced at the several steps in the run, including available logs, to find the point 
of failure. For assistance, you can open/search for issues in the 
[GitHub repository](https://github.com/nasa/ExoMiner/issues). You can also contact the 
[developers](https://github.com/nasa/ExoMiner). If you would like to share information about the run, please submit 
the run folder as a compressed file (include the inputs used). Be sure to exclude large files, such as the light curve 
FITS files and the DV XML files, and include the logs. **Keep your pipeline up-to-date by checking newer versions of 
the pipeline! You can watch the repository to keep track of all new activity including updates.**
