# mtil: multi-task imitation learning algorithms

This repository contains multi-task imitation learning baselines for use with
the MAGICAL benchmark suite. Currently it implements GAIL and behavioural
cloning. You can install the code with pip:

```sh
cd /path/to/this/directory
pip install -e .
sudo apt install xvfb  # necessary to run MAGICAL envs on a server
```

See notes below for important scripts, particularly the commands at the bottom
of this file that allow you to reproduce the results in the MAGICAL paper!

## Important parts of this repository

- `etc/`: this directory contains scripts and configurations for running
  experiments, post-processing data, etc.
  - `etc/comparison_runs.py`: script that takes in a YAML experiment spec and
    runs the corresponding experiments. There are several example experiment
    specs in the `etc/` directory.
  - `etc/expt-full.yml`: YAML experiment spec for the full set of experiments
    featured in the MAGICAL paper.
  - `etc/collate-comparison-runs.ipynb`: Jupyter notebook that can take the huge
    amount of data produced by `comparison_runs.py` and distil it down into a
    single table.
- `mtil/`: the main package containing implementations of all algorithms.
  - `mtil/algos/mtgail`: multi-task GAIL implementation. Normally this is
    invoked through `comparison_runs.py`. If you want to run it manually, use
    `python -m mtil.algos.mtgail --help` to see available options.
  - `mtil/algos/mtbc`: multi-task BC implementation. Again, this is normally
    executed by `comparison_runs.py`, but can also be executed manually. To see
    the options for training MTBC manually, see `python -m mtil.algos.mtbc train
    --help`. The MTBC implementation also contains code for testing models
    produced by both GAIL and BC. `python -m mtil.algos.mtbc testall` can be
    used to generate score statistics for a saved BC or GAIL policy on all
    applicable variants, while `python -m mtil.algos.mtbc test` can be used to
    view policy rollouts.

## Running a full set of experiments

Reproducing the experiments in the MAGICAL paper requires two steps:

1. First, you must use `etc/comparison_runs.py` to train and test all of the
   baselines.
2. Second, you must use `etc/collate-comparison-runs.ipynb` to collate the data
   produced by `comparison_runs.py` into a single table.
   
To perform the first step, you can use a command like the following:

```sh
xvfb-run -a python etc/comparison_runs.py --job-ngpus 0.25 --job-ngpus-eval 0.25 \
  --out-dir ./scratch/magical-repro/ etc/expt-full.yml
```

The two most important arguments here are `--job-ngpus` and `-job-ngpus-eval`,
respectively, which control what fraction of a GPU is reserved for each training
and evaluation run, respectively. The most memory-intensive runs typically take
~2.5GB of VRAM, so the settings above, which dedicate a quarter of a GPU to each
run, are adequate if you have GTX 1080 Tis with ~12GB of VRAM. Those figures may
need to be increased if you have less VRAM.

Beyond GPU options, the other important option is `--out-dir`. Once
`comparison_runs.py` has finished (which might take a week or more, depending on
the performance of your machine), that directory will contain all of the results
that can be processed by `collate-comparison-runs.ipynb`. To process those
results, open `etc/collate-comparison-runs.ipynb` in Jupyter, and set the
`CSV_PATTERN` variable in the second code cell to point to the location of the
`eval-*.csv` files produced by `comparison_runs.py`. For the invocation above,
the appropriate setting would be:

```python
CSV_PATTERN = '../scratch/magical-repro/run*/eval*.csv'
```

Once you've set `CSV_PATTERN`, you can run the rest of the notebook and it
should output LaTeX tables containing all of the desired results.
