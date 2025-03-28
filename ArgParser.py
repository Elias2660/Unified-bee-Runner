import argparse
"""
ArgParser.py

This module is responsible for parsing command-line arguments for the pipeline script. It uses the argparse library to define and handle various parameters required for different stages of the pipeline.

Functions:
    - get_args: Parses and returns the command-line arguments.

Command-line Arguments:
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
                            (dataset creation) Number of frames to buffer at the end of the video, default 0, NOTE/TODO: only works for each frame one class and time testing, not base version
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

Usage:
    This script is intended to be used as part of a larger pipeline. It is typically invoked from the command line or another script, such as master_run.py.

Example:
    python ArgParser.py --data_path /path/to/data --start 0 --end 6 --width 960 --height 720

"""


def get_args():
    description = """
    Runs the pipeline that processes the data using the specified model.

    This program expects the log files to be named in the form logNo.txt, logPos.txt, logNeg.txt.

    This script automatically converts the videos to .mp4 format and then runs the pipeline on the data. The video type can either be mp4 or h264.
    """

    poem = """
    One workflow to rule them all,
    one workflow to find them,
    One workflow to bring them all,
    and in the data directory train them;
    In the Land of ilab where the shadows lie.
    """

    # truncating the log file

    parser = argparse.ArgumentParser(description=description, epilog=poem)
    parser.add_argument(
        "--data_path",
        type=str,
        help='Path to the data, default "."',
        default=".",
        required=False,
    )
    parser.add_argument(
        "--start",
        type=int,
        help="(unifier)Start the pipeline at the given step, default 0",
        default=0,
        required=False,
    )
    parser.add_argument(
        "--end",
        type=int,
        help=
        "(unifier)end the pipeline at the given step, default 6 (will not stop)",
        default=6,
        required=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=
        "(unifier)Print debug information, activates debug for logger (and other scripts), default=False",
        default=False,
    )

    # BACKGROUND SUBTRACTION
    parser.add_argument(
        "--background-subtraction-type",
        choices=["MOG2", "KNN"],
        required=False,
        default=None,
        type=str,
        help=
        "(background subtraction) Background subtraction type to use, default None, you can either choose MOG2 or KNN",
    )
    # for make_validation_training
    parser.add_argument(
        "--width",
        type=int,
        help="(splitting the data) Width of the images, default 960",
        default=960,
        required=False,
    )
    parser.add_argument(
        "--height",
        type=int,
        help="(splitting the data) Height of the images, default 720",
        default=720,
        required=False,
    )

    # for sampling
    parser.add_argument(
        "--number-of-samples",
        type=int,
        help=
        "(sampling)the number of samples max that will be gathered by the sampler, default=40000",
        default=40000,
    )
    parser.add_argument(
        "--max-workers-video-sampling",
        type=int,
        help=
        "(sampling)The number of workers to use for the multiprocessing of the sampler, default=3",
        default=3,
    )
    parser.add_argument(
        "--frames-per-sample",
        type=int,
        help=
        "(sampling, splitting the data)The number of frames per sample, default=1",
        default=1,
    )
    parser.add_argument(
        "--normalize",
        type=bool,
        help="(sampling) normalize the images, default=True",
        default=True,
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        help="(sampling) The number of output channels, default=1",
        default=1,
    )
    parser.add_argument(
        "--k",
        type=int,
        help=
        "(making the splits) Number of folds for cross validation, default 3",
        default=3,
        required=False,
    )
    parser.add_argument(
        "--model",
        choices=[
            "alexnet",
            "resnet18",
            "resnet34",
            "bennet",
            "resnext50",
            "resnext34",
            "resnext18",
            "convnextxt",
            "convnextt",
            "convnexts",
            "convnextb",
        ],
        type=str,
        help="(making the splits) model to use, default alexnet",
        default="alexnet",
        required=False,
    )

    # CREATING THE DATASET
    parser.add_argument(
        "--fps",
        type=int,
        help="(dataset creation) frames per second, default 25",
        default=25,
    )
    parser.add_argument(
        "--starting-frame",
        type=int,
        help="(dataset creation) starting frame, default 1",
        default=1,
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        help="(dataset creation) space between frames, default 0",
        default=0,
    )
    parser.add_argument(
        "--test-by-time",
        action="store_true",
        required=False,
        default=False,
        help=
        "(dataset creation) creates a dataset.csv that is time based, rather than based on the log files. This is to test for correlation between daytime and class accuracy",
    )
    parser.add_argument(
        "--time-splits",
        type=int,
        default=3,
        required=False,
        help=
        "(dataset creation) if the --test-by-time option is called, this will determine the number of splits that will occur",
    )
    parser.add_argument(
        "--files",
        type=str,
        help=
        "(dataset creation) name of the log files that one wants to use, default logNo.txt, logNeg.txt, logPos.txt",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--each-video-one-class",
        help=
        "(dataset creation) the case where each video is one class; a special workflow",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--end-frame-buffer",
        type=int,
        default=0,
        help=
        "(dataset creation) the number of frames to buffer at the end of the video (NOTE/TODO: only works for each frame one class and time testing, not base version), default=0",
        required=False,
    )

    # SPLITTING UP THE DATASET

    parser.add_argument(
        "--seed",
        type=str,
        default="01011970",
        help=
        "(making the splits) Seed to use for randomizing the data sets, default: 01011970",
    )
    parser.add_argument(
        "--only_split",
        required=False,
        default=False,
        action="store_true",
        help=
        "(making the splits) Set to finish after splitting the csv, default: False",
    )
    parser.add_argument(
        "--crop_x_offset",
        type=int,
        required=False,
        default=0,
        help=
        "(making the splits) The offset (in pixels) of the crop location on the original image in the x dimension, default 0",
    )
    parser.add_argument(
        "--crop_y_offset",
        type=int,
        required=False,
        default=0,
        help=
        "(making the splits) The offset (in pixels) of the crop location on the original image in the y dimension, default 0",
    )
    parser.add_argument(
        "--training_only",
        type=bool,
        required=False,
        default=False,
        help=
        "(making the splits) only generate the training set files, default: False",
    )

    # FRAME COUNTING
    parser.add_argument(
        "--optimize-counting",
        action="store_true",
        help="(frame counting) optimize frame counting for .mp4, default=False",
        default=False,
    )
    parser.add_argument(
        "--max-workers-frame-counter",
        type=int,
        help=
        "(frame counting) The number of workers to use for the multiprocessing of the frame counter, default=20",
        default=20,
        required=False,
    )

    # BACKGROUND SUBTRACTION
    parser.add_argument(
        "--max-workers-background-subtraction",
        type=int,
        help=
        "(background subtraction) The number of workers to use for the multiprocessing of the background subtraction, default=10",
        default=10,
        required=False,
    )

    # TRAINING
    parser.add_argument(
        "--epochs",
        type=int,
        help="(training) The number of epochs to train the model, default=10",
        default=10,
        required=False,
    )
    parser.add_argument(
        "--gpus",
        type=int,
        required=False,
        help="(training) The number of gpus to use for training, default=1",
        default=1,
    )

    # for binary optimization
    parser.add_argument(
        "--binary-training-optimization",
        action="store_true",
        help="(training) Convert and train with binary files",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--use-dataloader-workers",
        action="store_true",
        default=False,
        required=False,
        help=
        "(training) Whether to use dataloader workers in the training script.",
    )

    parser.add_argument(
        "--max-dataloader-workers",
        type=int,
        default=3,
        required=False,
        help="(training) The number of dataloader workers, default=3",
    )

    parser.add_argument(
        "--loss-fn",
        type=str,
        default="CrossEntropyLoss",
        choices=[
            "NLLLoss",
            "BCEWithLogitsLoss",
            "CrossEntropyLoss",
            "L1Loss",
            "MSELoss",
            "BCELoss",
        ],
        required=False,
        help="(training) the loss function used for training",
    )

    parser.add_argument(
        "--gradcam-cnn-model-layer",
        nargs="+",
        required=False,
        choices=[
            "model_a.0.0",
            "model_a.1.0",
            "model_a.2.0",
            "model_a.3.0",
            "model_a.4.0",
            "model_b.0.0",
            "model_b.1.0",
            "model_b.2.0",
            "model_b.3.0",
            "model_b.4.0",
        ],
        default=["model_a.4.0", "model_b.4.0"],
        help=
        "(training, make validation training) Model layers for gradcam plots, default=['model_a.4.0', 'model_b.4.0']",
    )

    # SAMPLING

    parser.add_argument(
        "--crop",
        action="store_true",
        help="(sampling) Crop the images to the correct size",
        default=False,
    )
    parser.add_argument(
        "--equalize-samples",
        action="store_true",
        help=
        "(sampling) Equalize the samples so that each class has the same number of samples",
        default=False,
    )
    parser.add_argument(
        "--dataset-writing-batch-size",
        type=int,
        help="(sampling) The batch size for writing the dataset, default=10",
        default=30,
        required=False,
    )
    parser.add_argument(
        "--max-threads-pic-saving",
        type=int,
        help=
        "(sampling) The number of threads to use for saving the pictures, default=4",
        required=False,
        default=4,
    )
    parser.add_argument(
        "--max-batch-size-sampling",
        type=int,
        default=5,
        help=
        "(sampling) The maximum batch size for sampling the video, default=5",
    )
    parser.add_argument(
        "--max-workers-tar-writing",
        type=int,
        help=
        "(sampling) The number of workers to use for writing the tar files, default=4",
        required=False,
        default=4,
    )
    parser.add_argument(
        "--y-offset",
        type=int,
        help="The y offset for the crop, default=0",
        default=0,
    )
    parser.add_argument(
        "--out-width",
        type=int,
        help="The width of the output image, default=400",
        default=400,
    )
    parser.add_argument(
        "--out-height",
        type=int,
        help="The height of the output image, default=400",
        default=400,
    )

    args = parser.parse_args()
    return args
