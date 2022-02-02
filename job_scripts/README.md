# Jobs scripts

These scripts are used to submit jobs on NASA Ames High-End Computing Cluster (HECC). More information on the cluster
about access, resources and features can be found in the HECC's
[Knowledgebase](https://www.nas.nasa.gov/hecc/support/kb/).

### General structure of a PBS script

The HECC uses a PBS scheduling system to organize and prioritize jobs from the multiple HECC's users. These scripts have
following general structure. Commands interpreted by PBS start with `#PBS`.

```shell
#PBS -S /bin/bash
#PBS -N job_name
# time allocated to the job; different queues have different maximum time for allocation
#PBS -l walltime=08:00:00
# resources allocated to the job
# select: number of chunks allocated to the job.
# ncpus: number of CPUs allocated in a given chunk
# mpiprocs: number of MPI processes per chunk
# model: type of of node; different nodes have different specifications (number of GPUs, memory, CPUs, ...)
#  the type of nodes available will depend on the queue
# ngpus: number of GPUs allocated in a given chunk
# mem: memory  allocated in a given chunk
#PBS -l select=1:ncpus=36:mpiprocs=4:model=sky_gpu:ngpus=4:mem=360g
# for some queues, the place statement is also needed; check KB for more information
#PBS -l place=scatter:exclhost
# queue job is submitted to. Check KB for more information on the types of queues on the HECC.
# examples of other queues: v100, k40 (GPU queues); long (for long jobs); debug (for debugging).
#PBS -q dsg_gpu@pbspl4
# file path to files that save job's output and error logs from PBS (not output from user's code)
#PBS -o /home6/msaragoc/jobs/Kepler-TESS_exoplanet/job_cv_mgpus.out
#PBS -e /home6/msaragoc/jobs/Kepler-TESS_exoplanet/job_cv_mgpus.err
# group to which the job is charged to; must be a group that the user belongs to.
#PBS -W group_list=a1509
# user receives emails when the job starts and ends
#PBS -m bea

# Code to be executed
```