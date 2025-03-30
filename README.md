# Unified Bee Runner

Unified Bee Runner is a pipeline for processing and analyzing bee-related datasets. This project includes several steps such as dataset creation, video conversion, data splitting, and model training. The model is meant to run on ilab, and through slurm, and for the Behavioral Analysis project at [Rutgers WINLAB](https://www.winlab.rutgers.edu/).

This is mean to unify all the code that has been created for the [project (2024 presentation attached)](https://docs.google.com/presentation/d/1j25c85SZ_8YPYvNdfubfVlN3Zx5B-090/edit?usp=sharing&ouid=110217722607110726120&rtpof=true&sd=true).

## Pipeline

![PipelineImage](RunImage.png)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/Elias2660/Unified-bee-Runner.git
   ```

## Usage

To run the pipeline, use the provided [shell script](Unifier_Run.sh) in [SLURM](https://slurm.schedmd.com/documentation.html). If you don't want to, (this is not recommended though) you can execute it in your command line, though this will take a while:

**NOTE**: Do not `cd` into the `Unified-Bee-Runner` dir to run with default settings. They are configured to run in the dir containing the data. Your file structure should look like when you run the commands (you should be in data_dir):

```
data_dir
├── Unified_bee_Runner
├── d1.mp4
├── d2.mp4
├── d3.mp4
├── logNo.txt
└── logPos.txt
```

Run `squeue -u <user>` to be able to find your current jobs and the servers that they are running on.

You can edit the [`Unifier_Run.sh`](Unifier_Run.sh) file with the settings that you desire. You can check the settings by running [`python3 Unified-bee-Runner/master_run.py -h`](master_run.py) or checking in [`ArgParser.py`](ArgParser.py) for the arguments that can be used. Not all of them work, including those that crop, and this pipeline is still working through many bugs.

Then run:

```sh
sbatch -x [servers, such as server1,server2] Unified-bee-Runner/Unifier_Run.sh
```

**RECOMMENDED**: To run with default settings, you can run the [`Slurm_Run.sh`](Slurm_Run.sh) file, which has preset sbatch settings:

```sh
./Unified-bee-Runner/Slurm_Run.sh
```

## Pipeline Steps

This is run using the chapter system, so you can choose the specific steps that are used here by editing the `--start` and `--end` commands, which are by default respectively set at 0 and 6, which will run the entire model.

0. [`Video Conversion and Counting`](https://github.com/Elias2660/Video_Frame_Counter): Converts .h264 videos to .mp4 format and creates counts.csv.
1. [`Background Subtraction`](https://github.com/Elias2660/Video_Subtractions): Applies background subtraction to the video frames to isolate the bees
2. [`Dataset Creation`](https://github.com/Elias2660/Dataset_Creator): Creates the dataset.csv
3. [`Data Splitting`](https://github.com/Elias2660/working_bee_analysis/blob/main/make_validation_training.py): Splits the data into training and testing sets
4. [`Video Sampling`](https://github.com/Elias2660/VideoSamplerRewrite): Clones the VideoSamplerRewrite repository and samples the video frames
5. [`Model Training`](https://github.com/bfirner/bee_analysis/blob/main/VidActRecTrain.py): Runs the model training script

## Other Stuff

This projects has other project submodules in the [non_workflow_links](non_workflow_links/) directory, which relate to other parts of the bee project.

## Contributing

[Contributions](CONTRIBUTING.md) are welcome! Please open an issue or submit a pull request.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## [Security Policy](SECURITY.md)

### Reporting a Vulnerability

If you discover a security vulnerability within this project, please report it. We will respond to your report promptly and work with you to understand and address the issue.

## License

This project is licensed under the [MIT License](LICENSE).
