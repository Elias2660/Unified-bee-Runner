"""
Module Name: master_run.py

Description:
    Orchestrates the end‑to‑end processing pipeline for bee behavior videos on Rutgers CS infrastructure.
    Supports chapters for video conversion, background subtraction, dataset creation, data splitting,
    video sampling, and model training with k‑fold cross‑validation. Generates all intermediate CSVs,
    shell scripts, and SLURM submission scripts needed for preprocessing and training.

Usage:
    python master_run.py \
        --data-path <input_dir> \
        --out-path <output_dir> \
        [--start <chapter>] \
        [--end <chapter>] \
        [--debug] \
        [--background-subtraction-type {MOG2,KNN}] \
        [--width <px>] [--height <px>] \
        [--number-of-samples <n>] \
        [--max-workers-video-sampling <n>] \
        [--frames-per-sample <n>] \
        [--normalize <True|False>] \
        [--out-channels <n>] \
        [--k <folds>] \
        [--model <name>] \
        [--fps <fps>] \
        [--starting-frame <n>] \
        [--frame-interval <n>] \
        [--test-by-time] \
        [--time-splits <n>] \
        [--files <log1,log2,...>] \
        [--each-video-one-class] \
        [--end-frame-buffer <n>] \
        [--seed <value>] \
        [--only_split] \
        [--crop_x_offset <px>] [--crop_y_offset <px>] \
        [--training_only] \
        [--optimize-counting] \
        [--max-workers-frame-counter <n>] \
        [--max-workers-background-subtraction <n>] \
        [--epochs <n>] \
        [--gpus <n>] \
        [--binary-training-optimization] \
        [--use-dataloader-workers] \
        [--max-dataloader-workers <n>] \
        [--loss-fn <fn>] \
        [--gradcam-cnn-model-layer <layer>...] \
        [--crop] \
        [--equalize-samples] \
        [--dataset-writing-batch-size <n>] \
        [--max-threads-pic-saving <n>] \
        [--max-batch-size-sampling <n>] \
        [--max-workers-tar-writing <n>] \
        [--out-width <px>] [--out-height <px>]

Chapters:
    0: Video Conversion (.h264→.mp4 & counts.csv)
    1: Background Subtraction (MOG2/KNN)
    2: Dataset Creation (Make_Dataset.py / time_based_division.py / one_class_runner.py)
    3: Data Splitting (k‑fold CSVs & sbatch scripts)
    4: Video Sampling (frame extraction & tar creation)
    5: Model Training (SLURM jobs, optional binary conversion)

Arguments:
    --data-path                 Base directory for all video/log files.
    --out-path                  Where all outputs, logs, and scripts are written.
    --start, --end              Pipeline range of chapters to run (inclusive).
    --debug                     Enable DEBUG‑level logging.
    --background-subtraction-type
                                Choose MOG2 or KNN for step 1.
    --width, --height           Image crop dimensions for splitting/training.
    --number-of-samples         Max frames to sample per video.
    --max-workers-video-sampling
                                Workers for frame sampling.
    --frames-per-sample         Frames extracted per sample.
    --normalize, --out-channels
                                Image normalization and channel settings.
    --k, --model                Folds and model name for cross‑validation.
    --fps, --starting-frame,
    --frame-interval, --end-frame-buffer
                                Controls dataset creation timing and buffers.
    --test-by-time, --time-splits,
    --files, --each-video-one-class
                                Variants for dataset CSV generation.
    --seed                      RNG seed for reproducibility.
    --only_split, --training_only
                                Control split-only or training-only modes.
    --optimize-counting,
    --max-workers-frame-counter,
    --max-workers-background-subtraction
                                Parallelization flags for counting and subtraction.
    --epochs, --gpus, --binary-training-optimization,
    --use-dataloader-workers, --max-dataloader-workers, --loss-fn,
    --gradcam-cnn-model-layer
                                Training configuration options.
    --crop, --equalize-samples,
    --dataset-writing-batch-size,
    --max-threads-pic-saving,
    --max-batch-size-sampling,
    --max-workers-tar-writing,
    --out-width, --out-height
                                Sampling and dataset write options.

Logging:
    Uses Python’s logging module with time‑stamped INFO/DEBUG messages.
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

try:
    args = get_args()

    with open(os.path.join(args.out_path, "RUN_DESCRIPTION.log"), "w+") as rd:
        rd.write(
            f"start-is: {format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n"
        )
        rd.write(f"path: {DIR_NAME}\n")

        user = getpass.getuser()
        rd.write(f"User: {user}\n")

        branch = (subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=DIR_NAME).decode("utf-8").strip())

        rd.write(f"branch of Unified-bee-Runner: {branch}\n")

        commit = (subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=DIR_NAME).decode("utf-8").strip())
        rd.write(f"Version / Commit: {commit}\n")

        python_version = subprocess.check_output(["which",
                                                  "python3"]).decode().strip()
        rd.write(f"Python Version: {python_version}\n")

        system_info = subprocess.check_output(["uname", "-a"]).decode().strip()
        rd.write(f"system: {system_info}\n")
    logging.info("---- Starting the pipeline ----")

    logging.info("---- Upgrading pip ----")
    subprocess.run("pip install --upgrade pip >> /dev/null",
                   shell=True,
                   executable="/bin/bash")

    logging.info("---- Attempting to Protect Dataset ----")
    # the txt files are the log*.txt files, better safe than sorry
    # add some protections to prevent deletions of data
    # of course, it'll be better with more backups
    protected_file_list = [
        file for file in os.listdir(args.in_path)
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
        f"pip install --no-compile -r {os.path.join(DIR_NAME, 'requirements.txt')} >> /dev/null",
        shell=True,
        executable="/bin/bash",
    )

    logging.info("(0) Starting the pipeline")
except Exception as e:
    logging.error(f"Error: {e}")
    raise ValueError("Something went wrong in the beginning")

# write the information into the description, to be stored later as part of the results
with open(os.path.join(args.out_path, "RUN_DESCRIPTION.log"), "a") as run_desc:
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
    run_desc.write(f"Optimize Counting: {args.optimize_counting}\n")
    run_desc.write(f"Use .bin Files: {args.binary_training_optimization}\n")
    run_desc.write(f"Use dataloader workers: {args.use_dataloader_workers}\n")

    # path stuff
    run_desc.write(f"Data Path: {args.in_path}\n")
    run_desc.write(f"Out Path: {args.out_path}\n")

with open(os.path.join(args.out_path, "RUN_DESCRIPTION.log"), "a") as run_desc:
    run_desc.write("\n-- Parsed Arguments --\n")
    run_desc.write(f"Parsed Arguments: {args}\n")

with open(os.path.join(args.out_path, "RUN_DESCRIPTION.log"), "a") as run_desc:
    run_desc.write("\n-- Miscellaneous Settings --\n")
    run_desc.write(f"Seed: {args.seed}\n")
    run_desc.write(f"Normalize: {args.normalize}\n")
    run_desc.write(f"Width: {args.width}\n")
    run_desc.write(f"Height: {args.height}\n")
    run_desc.write(f"Frames per Second: {args.fps}\n")

with open(os.path.join(args.out_path, "RUN_DESCRIPTION.log"), "a") as run_desc:
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

# stupid ilab umask makes files unreadable / undeletable / uneditable to everyone
# except the creators, which is great for school projects but not so
# good for shared research / data projects
subprocess.run(
    f"chmod -R 777 Unified-bee-Runner {os.path.join(args.out_path, '*.log')} venv >> /dev/null 2>&1",
    shell=True,
)

if args.start > args.end:
    raise ValueError("You can't have the start be higher than the end")

# -----  STEP 0: Generating the counts.csv  -----
# if the videos a .h264, convert to .mp4, else, just make a counts.csv
# if you have --optimize-counting flag passed, then the program will utilize
# .mp4 file's indexing feature to count frames without actually running through them
# this however will only happen with

if args.start <= 0 and args.end >= 0:
    logging.info(
        "(0) Starting the video conversions, always defaulting to .mp4")
    try:

        logging.debug(
            "(0) ---- Installing the requirements for the Video_Frame_Counter ----"
        )
        subprocess.run(
            f"pip install --no-compile -r {os.path.join(DIR_NAME, 'Video_Frame_Counter/requirements.txt')} >> /dev/null",
            shell=True,
            executable="/bin/bash",
        )

        file_list = os.listdir(args.in_path)
        logging.info(f"file_list: {file_list}")
        contains_h264 = any(".h264" in file for file in file_list)
        contains_mp4 = any(".mp4" in file for file in file_list)

        # for the path, the output .mp4 videos, if they converted them,
        # would appear in the --out-path directory
        # the counts file
        arguments = (f" --video-filepath {args.in_path} "
                     f" --output-filepath {args.out_path}"
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
                f"python3 {os.path.join(DIR_NAME, 'Video_Frame_Counter/h264tomp4.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
                shell=True,
            )
        elif contains_mp4:
            if args.optimize_counting:
                logging.info("Making a fast counts.csv")
                subprocess.run(
                    f"python3 {os.path.join(DIR_NAME, 'Video_Frame_Counter/optimized_make_counts.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
                    shell=True,
                )
            else:
                logging.info("No conversion needed, making counts.csv")
                subprocess.run(
                    f"python3 {os.path.join(DIR_NAME, 'Video_Frame_Counter/make_counts.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
                    shell=True,
                )
        else:
            raise ValueError(
                "Something went wrong with the file typing, as it seems that there are no .h264 or .mp4 files in the directory"
            )

        logging.info("(0) ---- Changing Permissions for the Repository----")
        subprocess.run(
            f"chmod 777 {os.path.join(args.out_path, 'counts.csv')} >> /dev/null 2>&1",
            shell=True,
        )
    except Exception as e:
        logging.error(f"Error: {e}")
        raise ValueError("Something went wrong in step 0")
else:
    logging.info(
        f"Skipping step 0, given the start ({args.start}) and end ({args.end}) values"
    )

# ----- STEP 1: Background subtraction -----
# perform background subtraction, not the best right now, and least used step
# Currently supported background subtract is MOG2 and KNN

if args.start <= 1 and args.end >= 1:
    logging.info("(1) Starting the background subtraction")
    try:
        if args.background_subtraction_type is not None:
            logging.info("(1) Starting the background subtraction")

            logging.info(
                "(1) ---- Installing the requirements for the Video_Subtractions ----"
            )
            subprocess.run(
                f"pip install --no-compile -r {os.path.join(DIR_NAME, 'Video_Subtractions/requirements.txt')} >> /dev/null",
                shell=True,
            )

            # if the data directory contains .h264 files, then the files have
            # been converted to .mp4 in the args.out_path directory
            contains_h264 = any(".h264" in file
                                for file in os.listdir(args.in_path))

            arguments = (
                f" --path {args.in_path if not contains_h264 else args.out_path} "
                f" --dest-dir {os.path.join(args.out_path, 'unsubtracted_videos')}"
                f" --subtractor {args.background_subtraction_type} "
                f" --max-workers {args.max_workers_background_subtraction}")
            subprocess.run(
                f"python3 {os.path.join(DIR_NAME, 'Video_Subtractions/Convert.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
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
            f"pip install --no-compile -r {os.path.join(DIR_NAME, 'Dataset_Creator/requirements.txt')} >> /dev/null",
            shell=True,
        )
        if args.test_by_time:
            logging.info(
                "Deciding the test by time, given the passing of the --test-by-time button"
            )
            arguments = (
                f" --in-path {args.in_path} "
                f" --out-path {args.out_path}"
                # adding these options in case they need to be changed in the future
                f" --counts counts.csv "
                f" --start-frame {args.starting_frame} "
                f" --end-frame-buffer {args.end_frame_buffer} "
                f" --splits {args.time_splits} ")
            subprocess.run(
                f"python3 {os.path.join(DIR_NAME, 'Dataset_Creator/time_based_division.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
                shell=True,
            )

            # outputs equals the number of splits we divide the time
            num_outputs = args.time_splits

        elif args.each_video_one_class:
            logging.info(
                "Creating a one class dataset, given the passing of the --end-frame-buffer argument"
            )
            arguments = (
                f" --in-path {args.in_path} "
                f" --out-path {args.out_path}"
                # adding these options in case they need to be changed in the future
                f" --counts counts.csv "
                f" --start-frame {args.starting_frame} "
                f" --end-frame-buffer {args.end_frame_buffer} "
                f" --splits {args.k} ")
            subprocess.run(
                f"python3 {os.path.join(DIR_NAME, 'Dataset_Creator/one_class_runner.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
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
                f" --in-path {args.in_path} "
                f" --out-path {args.out_path} "
                # adding these options in case they need to be changed in the future
                f" --counts counts.csv "  # !just the name of the counts file, not the path
                f" --files '{string_log_list}' "
                f" --starting-frame {args.starting_frame} "
                f" --frame-interval {args.frame_interval} "
                f" --end-frame-buffer {args.end_frame_buffer} ")
            subprocess.run(
                f"python3 {os.path.join(DIR_NAME, 'Dataset_Creator/Make_Dataset.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
                shell=True,
            )

            # the number of outputs are equal the number of logs
            num_outputs = len(log_list)

        # changing the perms for the created dataset*.csv files
        logging.info("Changing the permissions for the created files")
        subprocess.run(
            f"chmod -R 777 {os.path.join(args.out_path, 'dataset*.csv')} {os.path.join(args.out_path, '*.bak')} >> /dev/null 2>&1",
            shell=True,
        )
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
            f"pip install --no-compile -r {os.path.join(DIR_NAME, 'bee_analysis/requirements.txt')} >> /dev/null",
            shell=True,
        )

        arguments = (
            f" --in-path {args.in_path} "
            f" --out-path {args.out_path} "
            f" --k {args.k} "
            f" --model {args.model} "
            f" --gpus {args.gpus} "
            f" --venv-path {os.path.join(DIR_NAME, '..')} "
            f" --seed {args.seed} "
            f" --width {args.width if not args.crop else args.out_width} "
            f" --height {args.height if not args.crop else args.out_height} "
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
            f"python3 {os.path.join(DIR_NAME, 'bee_analysis/make_validation_training.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
            shell=True,
        )

        logging.info("Changing permissions for the created files")
        subprocess.run(
            f"chmod -R 777 {os.path.join(args.out_path, 'dataset*.csv')} {os.path.join(args.out_path, '*.log')} {os.path.join(args.out_path, '*.sh')} >> /dev/null 2>&1",
            shell=True,
        )
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

        arguments = f" --in-path {args.out_path} " f" --out-path {args.out_path} "
        subprocess.run(
            f"python3 {os.path.join(DIR_NAME, 'Dataset_Creator/dataset_checker.py')} {arguments}",
            shell=True,
        )
        logging.info(
            "(4) ---- Installing the requirements for the VideoSamplerRewrite")
        subprocess.run(
            f"pip install --no-compile -r {os.path.join(DIR_NAME, 'VideoSamplerRewrite/requirements.txt')} >> /dev/null",
            shell=True,
        )
        contains_h264 = any(".h264" in file
                            for file in os.listdir(args.in_path))

        arguments = (
            f" --video-input-path  {args.out_path if contains_h264 else args.in_path} "
            f" --dataset-input-path {args.out_path} "
            f" --out-path {args.out_path} "
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
            arguments += (
                f" --crop --x-offset {args.crop_x_offset} "
                f" --y-offset {args.crop_y_offset} "
                f" --out-width {args.width if not args.crop else args.out_width} "
                f" --out-height {args.height if not args.crop else args.out_height}"
                f" --scale-factor {args.scale_factor} ")
        if args.debug:
            arguments += " --debug "
        if args.equalize_samples:
            arguments += " --equalize-samples "
        subprocess.run(
            f"python3 {os.path.join(DIR_NAME, 'VideoSamplerRewrite/Dataprep.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
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
            f" {os.path.join(args.out_path, file)} "
            f" --entries {' '.join([f'{i}.png' for i in range(args.frames_per_sample)])} cls "
            f" --handler_overrides cls stoi "
            f" --output {os.path.join(args.out_path, file.replace('tar', 'bin'))} "
            f" --shuffle {20000 // args.frames_per_sample} "
            f" --shardshuffle {20000 // args.frames_per_sample}")
        subprocess.run(
            f"python3 {os.path.join(DIR_NAME, 'bee_analysis/utility/webdataset_to_flatbin.py')} {arguments} >> {args.out_path}/dataprep.log 2>&1",
            shell=True,
        )

    if args.binary_training_optimization:
        logging.info(
            "(5) Creating .bin files given passing of --binary-training-optimization"
        )
        file_list = [
            file for file in os.listdir(args.out_path) if file.endswith(".tar")
        ]
        if len(file_list) == 0:
            raise Exception(
                "There are no tar files found. That means that there is an uncaught problem earlier in the pipeline that has caused there to be zero tar files. Try to debug that."
            )
        count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=min(int(count /
                                                      5), len(file_list)))
        pool.starmap(create_bin_file,
                     ((file, DIR_NAME, args) for file in file_list))
        logging.info("Bin files created.")

        # make sure that everyone can analyze these new files
        subprocess.run(
            f"chmod -R 777 {os.path.join(args.out_path, '*.bin')} >> /dev/null 2>&1",
            shell=True,
        )

    try:
        subprocess.run(
            f"chmod -R 777 Unified-bee-Runner  >> /dev/null 2>&1",
            shell=True,
        )
        subprocess.run(
            f"chmod 777 {args.out_path}  >> /dev/null 2>&1",
            shell=True,
        )
        os.chdir(args.out_path)
        subprocess.run(
            f"chmod -R 777 {os.path.join('*.log')} {os.path.join('*.sh')} >> /dev/null 2>&1",
            shell=True,
        )
        subprocess.run(f"./{os.path.join('training-run.sh')}", shell=True)
        logging.info("Submitted executors for training")
        logging.info("Pipeline complete")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise ValueError("Something went wrong in step 5")
else:
    logging.info(
        f"Skipping step 5, given the start ({args.start}) and end ({args.end}) values"
    )
