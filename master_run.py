"""

This script is organized in chapters, so you can use the start and end flags to run specific chapters. The chapters are as follows:

Arguments:
    -h, --help
                            Show this help message and exit.
    --data_path DATA_PATH
                            Path to the data, default is ".".
    --start START
                            (unifier) Start the pipeline at the given step, default 0.
    --end END
                            (unifier) End the pipeline at the given step, default 6 (will not stop).
    --debug
                            (unifier) Print debug information and activate debug mode for logger (and other scripts), default False.
    --background-subtraction-type {MOG2,KNN}
                            (background subtraction) Background subtraction type to use, default None. Options: MOG2 or KNN.
    --width WIDTH
                            (splitting the data) Width of the images, default 960.
    --height HEIGHT
                            (splitting the data) Height of the images, default 720.
    --number-of-samples NUMBER_OF_SAMPLES
                            (sampling) The maximum number of samples to be gathered by the sampler, default 40000.
    --max-workers-video-sampling MAX_WORKERS_VIDEO_SAMPLING
                            (sampling) The number of workers to use for multiprocessing of the sampler, default 3.
    --frames-per-sample FRAMES_PER_SAMPLE
                            (sampling, splitting the data) Number of frames per sample, default 1.
    --normalize NORMALIZE
                            (sampling) Normalize the images, default True.
    --out-channels OUT_CHANNELS
                            (sampling) The number of output channels, default 1.
    --k K
                            (making the splits) Number of folds for cross-validation, default 3.
    --model MODEL
                            (making the splits) Model to use, default "alexnet".
    --fps FPS
                            (dataset creation) Frames per second, default 25.
    --starting-frame STARTING_FRAME
                            (dataset creation) Starting frame, default 1.
    --frame-interval FRAME_INTERVAL
                            (dataset creation) Space between frames, default 0.
    --test-by-time
                            (dataset creation) Create a dataset.csv based on time (instead of log files) to test correlation between daytime and class accuracy.
    --time-splits TIME_SPLITS
                            (dataset creation) If --test-by-time is used, determine the number of splits to occur.
    --files FILES
                            (dataset creation) Name of the log files to use, default "logNo.txt, logNeg.txt, logPos.txt".
    --each-video-one-class
                            (dataset creation) Treat each video as one class; a special workflow.
    --end-frame-buffer END_FRAME_BUFFER
                            (dataset creation) Number of frames to buffer at the end of the video, default 0
    --seed SEED
                            (making the splits) Seed for randomizing the data sets, default "01011970".
    --only_split
                            (making the splits) Finish after splitting the CSV, default False.
    --crop_x_offset CROP_X_OFFSET
                            (making the splits) Offset (in pixels) of the crop location on the original image in the x dimension, default 0.
    --crop_y_offset CROP_Y_OFFSET
                            (making the splits) Offset (in pixels) of the crop location on the original image in the y dimension, default 0.
    --training_only TRAINING_ONLY
                            (making the splits) Only generate the training set files, default False.
    --max-workers-frame-counter MAX_WORKERS_FRAME_COUNTER
                            (frame counting) Number of workers for multiprocessing of the frame counter, default 20.
    --max-workers-background-subtraction MAX_WORKERS_BACKGROUND_SUBTRACTION
                            (background subtraction) Number of workers for multiprocessing of background subtraction, default 10.
    --epochs EPOCHS
                            (training) Number of epochs to train the model, default 10.
    --gpus GPUS
                            (training) Number of GPUs to use for training, default 1.
    --gradcam-cnn-model-layer {model_a.0.0,model_a.1.0,model_a.2.0,model_a.3.0,model_a.4.0,model_b.0.0,model_b.1.0,model_b.2.0,model_b.3.0,model_b.4.0} [...]
                            (training, make validation training) Model layers for gradcam plots, default ['model_a.4.0', 'model_b.4.0'].
    --crop
                            (sampling) Crop the images to the correct size.
    --equalize-samples
                            (sampling) Equalize the samples so that each class has the same number of samples.
    --dataset-writing-batch-size DATASET_WRITING_BATCH_SIZE
                            (sampling) Batch size for writing the dataset, default 10.
    --max-threads-pic-saving MAX_THREADS_PIC_SAVING
                            (sampling) Number of threads to use for saving pictures, default 4.
    --max-batch-size-sampling MAX_BATCH_SIZE_SAMPLING
                            (sampling) Maximum batch size for sampling the video, default 5.
    --max-workers-tar-writing MAX_WORKERS_TAR_WRITING
                            (sampling) Number of workers for writing tar files, default 4.
    --y-offset Y_OFFSET
                            Y offset for the crop, default 0.
    --out-width OUT_WIDTH
                            Width of the output image, default 400.
    --out-height OUT_HEIGHT
                            Height of the output image, default 400.


Logging:
The script uses logging to provide information about the progress and any errors encountered during the execution of the pipeline.
Unified Bee Runner Pipeline Script

This script orchestrates the entire pipeline for processing and analyzing bee-related datasets. The pipeline is divided into several steps, each performing a specific task. The steps can be controlled using the `--start` and `--end` arguments, allowing users to run specific parts of the pipeline.

Chapter System:
This script is organized in chapters, so you can use the start and end flags to run specific chapters. The chapters are as follows:
0. Video Conversions
1. Background Subtraction
2. Dataset Creation
3. Data Splitting
4. Video Sampling
5. Model Training -> this will create slurm jobs given the number of k-folds that you have requested

- `--start`: The step with which to start the pipeline.
- `--end`: The step with which to end the pipeline.
"""
import getpass
import logging
import multiprocessing
import os
import subprocess
from datetime import datetime
from stat import S_IREAD
from stat import S_IRGRP
from stat import S_IROTH

from ArgParser import get_args

logging.basicConfig(format="%(asctime)s: %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")

DIR_NAME = os.path.dirname(os.path.abspath(__file__))

with open("RUN_DESCRIPTION.log", "w+") as rd:
    rd.write(
        f"start-is: {format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")
    rd.write(f"path: {DIR_NAME}\n")

    user = getpass.getuser()
    rd.write(f"User: {user}\n")

    branch = (subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=DIR_NAME).decode("utf-8").strip())

    rd.write(f"branch of Unified-bee-Runner: {branch}\n")

    commit = (subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      cwd=DIR_NAME).decode("utf-8").strip())
    rd.write(f"Version / Commit: {commit}\n")

try:
    args = get_args()
    logging.info("---- Starting the pipeline ----")
    path = args.data_path
    os.chdir(path)
    logging.info("---- Purging all packages ----")
    subprocess.run("xargs pip uninstall -y >> /dev/null",
                   shell=True,
                   executable="/bin/bash")

    logging.info("---- Upgrading pip ----")
    subprocess.run("pip install --upgrade pip >> /dev/null",
                   shell=True,
                   executable="/bin/bash")

    logging.info("---- Attempting to Protect Dataset ----")
    # the txt files are the log*.txt files, better safe than sorry
    # add some protections to prevent deletions of data
    # of course, it'll be better with more backups
    protected_file_list = [
        file for file in os.listdir()
        if (file.endswith(".txt") or file.endswith(".mp4")
            or file.endswith(".txt"))
    ]

    for file in protected_file_list:
        try:
            os.chmod(file, S_IREAD | S_IRGRP | S_IROTH)
        except:
            pass

    logging.info("---- Installing some requirements for the pipeline ----")
    subprocess.run(
        f"pip install -r {os.path.join(DIR_NAME, 'requirements.txt')} >> /dev/null",
        shell=True,
        executable="/bin/bash",
    )
    file_list = os.listdir()
    logging.info("(0) Starting the pipeline")
except Exception as e:
    logging.error(f"Error: {e}")
    raise ValueError("Something went wrong in the beginning")

# write the information into the description, for readability
with open("RUN_DESCRIPTION.log", "a") as run_desc:
    run_desc.write("\n-- Run Settings --\n")
    run_desc.write(f"Attempted Samples Per Video: {args.number_of_samples}\n")
    run_desc.write(f"Frames per Sample: {args.frames_per_sample}\n")
    run_desc.write(f"Equalized: {args.equalize_samples}\n")
    run_desc.write(
        f"Background subtraction: {args.background_subtraction_type}\n")
    run_desc.write(f"Model: {args.model}\n")
    run_desc.write(f"Epochs: {args.epochs}\n")
    run_desc.write(f"Crop: {args.crop}\n")
    run_desc.write(f"K-Splits: {args.k}\n")
    run_desc.write(f"START: {args.start}\n")
    run_desc.write(f"END: {args.end}\n")

with open("RUN_DESCRIPTION.log", "a") as run_desc:
    run_desc.write("\n-- Miscellaneous Settings --\n")
    run_desc.write(f"Seed: {args.seed}\n")
    run_desc.write(f"Normalize: {args.normalize}\n")
    run_desc.write(f"Width: {args.width}\n")
    run_desc.write(f"Height: {args.height}\n")
    run_desc.write(f"Frames per Second: {args.fps}\n")

with open("RUN_DESCRIPTION.log", "a") as run_desc:
    run_desc.write("\n-- Additional Options --\n")
    run_desc.write(f"Crop Enabled: {args.crop}\n")
    if args.crop:
        run_desc.write(f"Crop x: {args.crop_x_offset}\n")
        run_desc.write(f"Crop y: {args.crop_y_offset}\n")
        run_desc.write(f"Out width: {args.out_width}\n")
        run_desc.write(f"Out height: {args.out_height}\n")
    run_desc.write("\n-- Testing Modes --\n")
    run_desc.write(f"Debug Mode: {args.debug}\n")
    run_desc.write(f"Testing by Time: {args.test_by_time}\n")
    run_desc.write(f"Each Video One Class: {args.each_video_one_class}\n")
    run_desc.write("\n-- Other Modes --\n")
    run_desc.write(f"Training Only: {args.training_only}\n")

# convert the video
# At this point, only the Unified_Bee_Runner, Dataprep.log, venv, and RUN_DESCRIPTION.log need to be modified

# stupid ilab umask makes files unreadable / undeletable to everyone
# except the creators, which is great for school projects but not so
# good for shared research / data projects
subprocess.run("chmod -R 777 Unified-bee-Runner *.log venv >> /dev/null 2>&1",
               shell=True)

if args.start > args.end:
    raise ValueError("You can't have the start be higher than the end")

# -----  STEP 0: Generating the counts.csv  -----
# if the videos a .h264, convert to .mp4, else, just make a counts.csv

if args.start <= 0 and args.end >= 0:
    logging.info(
        "(0) Starting the video conversions, always defaulting to .mp4")
    try:

        logging.debug(
            "(0) ---- Installing the requirements for the Video_Frame_Counter ----"
        )
        subprocess.run(
            f"pip install -r {os.path.join(DIR_NAME, 'Video_Frame_Counter/requirements.txt')} >> /dev/null",
            shell=True,
            executable="/bin/bash",
        )

        file_list = os.listdir(path)
        contains_h264 = any(".h264" in file for file in file_list)
        contains_mp4 = any(".mp4" in file for file in file_list)

        arguments = (f" --path {path} "
                     f" --max-workers {args.max_workers_frame_counter} ")

        logging.info("(0) ---- Running Video Conversions Sections ----")

        if contains_h264 and contains_mp4:
            raise ValueError(
                "Both types of file are in this directory, please remove one")
        elif contains_h264:
            logging.info(
                "Converting .h264 to .mp4, old h264 files can be found in the h264_files folder"
            )
            subprocess.run(
                f"python3 {os.path.join(DIR_NAME, 'Video_Frame_Counter/h264tomp4.py')} {arguments} >> dataprep.log 2>&1",
                shell=True,
            )
        elif contains_mp4:
            if args.optimize_counting:
                logging.info("Making a fast counts.csv")
                subprocess.run(
                    f"python3 {os.path.join(DIR_NAME, 'Video_Frame_Counter/optimized_make_counts.py')} {arguments} >> dataprep.log 2>&1",
                    shell=True,
                )
            else:
                logging.info("No conversion needed, making counts.csv")
                subprocess.run(
                    f"python3 {os.path.join(DIR_NAME, 'Video_Frame_Counter/make_counts.py')} {arguments} >> dataprep.log 2>&1",
                    shell=True,
                )
        else:
            raise ValueError(
                "Something went wrong with the file typing, as it seems that there are no .h264 or .mp4 files in the directory"
            )

        logging.info("(0) ---- Changing Permissions for the Repository----")
        subprocess.run("chmod 777 counts.csv >> /dev/null 2>&1", shell=True)
    except Exception as e:
        logging.error(f"Error: {e}")
        raise ValueError("Something went wrong in step 0")
else:
    logging.info(
        f"Skipping step 0, given the start ({args.start}) and end ({args.end}) values"
    )

# ----- STEP 1: Background subtraction -----
# perform background subtraction, not the best right now, and least used step

if args.start <= 1 and args.end >= 1:
    logging.info("(1) Starting the background subtraction")
    try:
        if args.background_subtraction_type is not None:
            logging.info("(1) Starting the background subtraction")

            logging.info(
                "(1) ---- Installing the requirements for the Video_Subtractions ----"
            )
            subprocess.run(
                f"pip install -r {os.path.join(DIR_NAME, 'Video_Subtractions/requirements.txt')} >> /dev/null",
                shell=True,
            )

            arguments = (
                f" --subtractor {args.background_subtraction_type} "
                f" --max-workers {args.max_workers_background_subtraction}")
            subprocess.run(
                f"python3 {os.path.join(DIR_NAME, 'Video_Subtractions/Convert.py')} {arguments} >> dataprep.log 2>&1",
                shell=True,
            )

        else:
            logging.info(
                "No background subtraction type given, skipping this step")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise ValueError("Something went wrong in step 1")
else:
    logging.info(
        f"Skipping step 1, given the start ({args.start}) and end ({args.end}) values"
    )

# this is for the num_outputs argument which will be used to tell the model how many classes there are
num_outputs = 0

# -----  STEP 2: Creating dataset.csv -----
# This the dataset.csv file relates the raw video to the log files, which then is
# used by the sampling in order to generate examples

if args.start <= 2 and args.end >= 2:
    logging.info("(2) Starting the dataset creation")
    try:

        logging.info(
            "(2) ---- Installing the requirements for the Dataset_Creator ----"
        )
        subprocess.run(
            f"pip install -r {os.path.join(DIR_NAME, 'Dataset_Creator/requirements.txt')} >> /dev/null",
            shell=True,
        )
        if args.test_by_time:
            logging.info(
                "Deciding the test by time, given the passing of the --test-by-time button"
            )
            arguments = (
                f" --path {path} "
                # adding these options in case they need to be changed in the future
                f" --counts counts.csv "
                f" --start-frame {args.starting_frame} "
                f" --end-frame-buffer {args.end_frame_buffer} "
                f" --splits {args.time_splits} ")
            subprocess.run(
                f"python3 {os.path.join(DIR_NAME, 'Dataset_Creator/time_based_division.py')} {arguments} >> dataprep.log 2>&1",
                shell=True,
            )

            # outputs equals the number of splits we divide the time
            num_outputs = args.time_splits

        elif args.each_video_one_class:
            logging.info(
                "Creating a one class dataset, given the passing of the --end-frame-buffer argument"
            )
            arguments = (
                f" --path {path} "
                # adding these options in case they need to be changed in the future
                f" --counts counts.csv "
                f" --start-frame {args.starting_frame} "
                f" --end-frame-buffer {args.end_frame_buffer} "
                f" --splits {args.k} ")
            subprocess.run(
                f"python3 {os.path.join(DIR_NAME, 'Dataset_Creator/one_class_runner.py')} {arguments} >> dataprep.log 2>&1",
                shell=True,
            )

            # outputs equal to the number of k splits
            num_outputs = args.k
        else:
            logging.info("(2) Creating a dataset.csv based on the txt files")

            log_list = [
                file for file in file_list
                if file.startswith("log") and file.endswith(".txt")
            ]

            logging.info(
                f"(2) Creating the dataset with the files: {log_list}")

            if args.files is None:
                string_log_list = ",".join(log_list).strip().replace(" ", "")
            else:
                string_log_list = args.files

            arguments = (
                f" --path {path} "
                # adding these options in case they need to be changed in the future
                f" --counts counts.csv "
                f" --files '{string_log_list}' "
                f" --starting-frame {args.starting_frame} "
                f" --frame-interval {args.frame_interval} "
                f" --end-frame-buffer {args.end_frame_buffer} ")
            subprocess.run(
                f"python3 {os.path.join(DIR_NAME, 'Dataset_Creator/Make_Dataset.py')} {arguments} >> dataprep.log 2>&1",
                shell=True,
            )

            # the number of outputs are equal the number of logs
            num_outputs = len(log_list)

        # changing the perms for the created dataset*.csv files
        logging.info("Changing the permissions for the created files")
        subprocess.run("chmod -R 777 dataset*.csv *.bak >> /dev/null 2>&1",
                       shell=True)
    except Exception as e:
        logging.error(f"Error: {e}")
        raise ValueError("Something went wrong in step 2")
else:
    logging.info(
        f"Skipping step 2, given the start ({args.start}) and end ({args.end}) values"
    )

# -----  STEP 3: Splitting up the data -----
# This creates the .sh files along with the dataset_*.csv files. This
# is the last step until the Video Sampler can create the tar files of
# samples

logging.info("(3) Splitting up the data")
if args.start <= 3 and args.end >= 3:
    try:
        logging.info("(3) Starting the data splitting")
        logging.info(
            "(3) ---- Installing the requirements for the bee_analysis ----")
        subprocess.run(
            f"pip install -r {os.path.join(DIR_NAME, 'bee_analysis/requirements.txt')} >> /dev/null",
            shell=True,
        )

        arguments = (
            f" --k {args.k} "
            f" --model {args.model} "
            f" --gpus {args.gpus} "
            f" --seed {args.seed} "
            f" --width {args.width} "
            f" --height {args.height} "
            f" --path_to_file {os.path.join(DIR_NAME, 'bee_analysis')} "
            f" --frames_per_sample {args.frames_per_sample} "
            f" --crop_x_offset {args.crop_x_offset} "
            f" --crop_y_offset {args.crop_y_offset} "
            f" --epochs {args.epochs} "
            f" --gradcam_cnn_model_layer {' '.join(args.gradcam_cnn_model_layer)} "
            # inferred from the Dataset Creation aspects of the workflow (see: step 2)
            f" --num-outputs {num_outputs} "
            f" --loss-fn {args.loss_fn} ")
        if args.only_split:
            arguments += " --only_split "
        if args.training_only:
            arguments += " --training_only "
        if args.each_video_one_class:
            arguments += " --remove-dataset-sub "
        if args.binary_training_optimization:
            arguments += " --binary-training-optimization "
        if args.use_dataloader_workers:
            arguments += (
                " --use-dataloader-workers "
                f" --max-dataloader-workers {args.max_dataloader_workers}")

        subprocess.run(
            f"python3 {os.path.join(DIR_NAME, 'bee_analysis/make_validation_training.py')} {arguments} >> dataprep.log 2>&1",
            shell=True,
        )

        logging.info("Changing permissions for the created files")
        subprocess.run(
            "chmod -R 777 dataset*.csv *.log *.sh >> /dev/null 2>&1",
            shell=True)
    except Exception as e:
        logging.error(f"Error: {e}")
        raise ValueError("Something went wrong in step 3")
else:
    logging.info(
        f"Skipping step 3, given the start ({args.start}) and end ({args.end}) values"
    )

# -----  STEP 4: Creating .tar files with samples -----
# Generate .npz and .txt temporary storage for samples, and then
# write that into a tar file

logging.info("(4) Starting the tar sampling")
if args.start <= 4 and args.end >= 4:
    try:
        logging.info("(4) Starting the video sampling")

        subprocess.run(
            f"python3 {os.path.join(DIR_NAME, 'Dataset_Creator/dataset_checker.py')}",
            shell=True,
        )
        logging.info(
            "(4) ---- Installing the requirements for the VideoSamplerRewrite")
        subprocess.run(
            f"pip install -r {os.path.join(DIR_NAME, 'VideoSamplerRewrite/requirements.txt')} >> /dev/null",
            shell=True,
        )

        arguments = (
            f" --dataset_path {path} "
            f" --frames-per-sample {args.frames_per_sample} "
            f" --number-of-samples {args.number_of_samples} "
            f" --normalize {args.normalize} "
            f" --out-channels {args.out_channels} "
            f" --max-workers {args.max_workers_video_sampling} "
            f" --dataset-writing-batch-size {args.dataset_writing_batch_size} "
            f" --max-threads-pic-saving {args.max_threads_pic_saving} "
            f" --max-workers-tar-writing {args.max_workers_tar_writing} "
            f" --max-batch-size-sampling {args.max_batch_size_sampling} ")
        if args.crop:
            arguments += (f" --crop --x-offset {args.crop_x_offset} "
                          f" --y-offset {args.crop_y_offset} "
                          f" --out-width {args.width} "
                          f" --out-height {args.height}")
        if args.debug:
            arguments += " --debug "
        if args.equalize_samples:
            arguments += " --equalize-samples "
        subprocess.run(
            f"python3 {os.path.join(DIR_NAME, 'VideoSamplerRewrite/Dataprep.py')} {arguments} >> dataprep.log 2>&1",
            shell=True,
        )

    except Exception as e:
        logging.error(f"Error: {e}")
        raise ValueError("Something went wrong in step 4")

else:
    logging.info(
        f"Skipping step 4, given the start ({args.start}) and end ({args.end}) values"
    )

# ----- STEP 5: Model Training -----
# Train the model on the given samples, and if optimized for binary training,
# convert the .tar files to .bin files

logging.info("(5) Starting the model training")

if args.start <= 5 and args.end >= 5:
    # shell function to pass for multiprocessing
    def create_bin_file(file, DIR_NAME, args):
        arguments = (
            f" {file} "
            f" --entries {' '.join([f'{i}.png' for i in range(args.frames_per_sample)])} cls "
            f" --handler_overrides cls stoi "
            f" --output {file.replace('tar', 'bin')} "
            f" --shuffle {20000 // args.frames_per_sample} "
            f" --shardshuffle {20000 // args.frames_per_sample}")
        subprocess.run(
            f"python3 {os.path.join(DIR_NAME, 'bee_analysis/utility/webdataset_to_flatbin.py')} {arguments} >> dataprep.log 2>&1",
            shell=True,
        )

    if args.binary_training_optimization:
        logging.info(
            "(5) Creating .bin files given passing of --binary-training-optimization"
        )
        file_list = [file for file in os.listdir() if file.endswith(".tar")]

        count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=min(int(count /
                                                      5), len(file_list)))
        pool.starmap(create_bin_file,
                     ((file, DIR_NAME, args) for file in file_list))
        logging.info("Bin files created.")

        # make sure that everyone can analyze these new files
        subprocess.run("chmod -R 777 *.bin >> /dev/null 2>&1", shell=True)

    try:
        subprocess.run(
            "chmod -R 777 *.log *.sh Unified-bee-Runner >> /dev/null 2>&1",
            shell=True)
        subprocess.run("./training-run.sh", shell=True)
        logging.info("Submitted executors for training")
        logging.info("Pipeline complete")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise ValueError("Something went wrong in step 5")
else:
    logging.info(
        f"Skipping step 5, given the start ({args.start}) and end ({args.end}) values"
    )
