# Getting started

This page provides instructions on setting up Podman, pulling the image for the ExoMiner Pipeline, and additional prerequisites for running it. For those users that want to more flexiblity and control over the pipeline, they can download the code available on this [GitHub repository](https://github.com/nasa/Exominer/tree/main).

## Installing Podman

To use the ExoMiner Pipeline image you need to download and install [Podman](https://podman.io). Further instructions can be found in [here](https://podman.io/docs/installation). Make sure that a Podman machine is running: ```podman machine list```.

## Pulling the image

To pull the latest image of the ExoMiner Pipeline from the GitHub registry, run the following command:

```bash
podman pull ghcr.io/migmartinho/exominer_pipeline:latest
```

Once you pull the image, you can find the image by running ```podman images```. It should output a list containing a Podman image with name `ghcr.io/migmartinho/exominer_pipeline` under `REPOSITORY`.

## Additional prerequisites

The following files can be found in the [exominer_pipeline](/exominer_pipeline/) module and complement the Podman image by working as an interface between the user and the pipeline:

- [run_podman_application.sh](/exominer_pipeline/run_podman_application.sh): this is the main script that you can run to use the podman image of the ExoMiner pipeline. It is a wrapper file for `podman run`.
- [pipeline_run_config.yaml](/exominer_pipeline/pipeline_run_config.yaml): defines the parameters of the run and it is used as an argument to the ExoMiner Pipeline application.

A stable internet connection is also required to access the TESS repositories at the MAST and Gaia DR2 data.

### More information on the image

You can get detailed information on the Podman image by running the command ```podman images inspect exominer_pipeline```. Run ```podman image history exominer_pipeline``` to see the list of layers and commands that were used to create them. If you want to access the contents of the image, you can run the command ```podman run --entrypoint="" -it exominer_pipeline /bin/bash```.