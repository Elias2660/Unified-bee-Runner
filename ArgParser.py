import argparse
"""
Module Name: ArgParser.py

Description:
    Defines and parses command‑line arguments for the unified bee runner pipeline.
    Covers all stages from video conversion through model training with k‑fold cross‑validation.
    Flags are organized by pipeline chapter and control behavior such as:
      • Video conversion (.h264→.mp4, frame counting)
      • Background subtraction (MOG2 or KNN)
      • Dataset creation (time‑based, log‑based, one‑class)
      • Data splitting (k‑fold, CSV generation)
      • Video sampling (frame extraction, tar packaging)
      • Model training (SLURM scripts, binary optimization, GradCAM)

Usage:
    from ArgParser import get_args
    args = get_args()

    # Or standalone:
    python ArgParser.py \
      --data-path <input_dir> \
      --out-path <output_dir> \
      --start 0 --end 5 \
      [--background-subtraction-type MOG2] \
      [--fps 25 --starting-frame 1 --frame-interval 0] \
      [--k 3 --model alexnet] \
      [--number-of-samples 40000 --frames-per-sample 1] \
      [--optimize-counting] \
      [--epochs 10 --gpus 1] \
      [--binary-training-optimization] \
      [--use-dataloader-workers --max-dataloader-workers 3] \
      [--loss-fn CrossEntropyLoss] \
      [--gradcam-cnn-model-layer model_a.4.0 model_b.4.0] \
      [--crop --out-width 400 --out-height 400] \
      [--equalize-samples] \
      [--debug]

Arguments:
    --data-path                   Base input directory (default: ".")
    --out-path                    Output directory for all generated files
    --start                       First pipeline chapter to run (0–5, default: 0)
    --end                         Last pipeline chapter to run (0–5, default: 6)
    --debug                       Enable DEBUG logging

    # Video Conversion / Frame Counting
    --optimize-counting           Fast .mp4 frame counting (uses metadata)
    --max-workers-frame-counter   Workers for frame counting (default: 20)

    # Background Subtraction
    --background-subtraction-type MOG2|KNN (default: None)
    --max-workers-background-subtraction Workers for parallel subtraction (default: 10)

    # Dataset Creation
    --fps                         Frames per second (default: 25)
    --starting-frame              Start frame offset (default: 1)
    --frame-interval              Frame interval between segments (default: 0)
    --end-frame-buffer            Frames to buffer at end (default: 0)
    --test-by-time                Time‑based dataset splitting
    --time-splits                 Number of time splits (default: 3)
    --files                       Comma‑separated log files list
    --each-video-one-class        Treat each video as its own class

    # Data Splitting
    --k                           Number of folds for cross‑validation (default: 3)
    --model                       Model name for training scripts (default: alexnet)
    --seed                        RNG seed for splitting (default: "01011970")
    --only_split                  Exit after CSV splitting
    --training_only               Generate only training scripts
    --crop_x_offset, --crop_y_offset  Crop offsets (px, default: 0)
    --width, --height             Image dimensions for splitting/training (default: 960×720)

    # Video Sampling
    --number-of-samples           Max samples per video (default: 40000)
    --max-workers-video-sampling  Workers for sampling (default: 3)
    --frames-per-sample           Frames per sample (default: 1)
    --normalize                   Normalize images (default: True)
    --out-channels                Output channels (default: 1)
    --crop                        Enable cropping to output size
    --out-width, --out-height     Crop dimensions (px, default: 400×400)
    --equalize-samples            Equalize class sample counts
    --dataset-writing-batch-size  Batch size for tar writing (default: 30)
    --max-threads-pic-saving      Threads for image saving (default: 4)
    --max-batch-size-sampling     Sample batch size (default: 5)
    --max-workers-tar-writing     Workers for tar writing (default: 4)

    # Training
    --epochs                      Number of training epochs (default: 10)
    --gpus                        GPUs per SLURM job (default: 1)
    --binary-training-optimization
                                  Convert and train with binary files
    --use-dataloader-workers      Enable DataLoader multiprocessing
    --max-dataloader-workers      Number of DataLoader workers (default: 3)
    --loss-fn                     Loss function choice (default: CrossEntropyLoss)
    --gradcam-cnn-model-layer     GradCAM layers list (default: model_a.4.0, model_b.4.0)
""" """
Module Name: ArgParser.py

Description:
    Defines and parses command‑line arguments for the unified bee runner pipeline.
    Covers all stages from video conversion through model training with k‑fold cross‑validation.
    Flags are organized by pipeline chapter and control behavior such as:
      • Video conversion (.h264→.mp4, frame counting)
      • Background subtraction (MOG2 or KNN)
      • Dataset creation (time‑based, log‑based, one‑class)
      • Data splitting (k‑fold, CSV generation)
      • Video sampling (frame extraction, tar packaging)
      • Model training (SLURM scripts, binary optimization, GradCAM)

Usage:
    from ArgParser import get_args
    args = get_args()

    # Or standalone:
    python ArgParser.py \
      --data-path <input_dir> \
      --out-path <output_dir> \
      --start 0 --end 5 \
      [--background-subtraction-type MOG2] \
      [--fps 25 --starting-frame 1 --frame-interval 0] \
      [--k 3 --model alexnet] \
      [--number-of-samples 40000 --frames-per-sample 1] \
      [--optimize-counting] \
      [--epochs 10 --gpus 1] \
      [--binary-training-optimization] \
      [--use-dataloader-workers --max-dataloader-workers 3] \
      [--loss-fn CrossEntropyLoss] \
      [--gradcam-cnn-model-layer model_a.4.0 model_b.4.0] \
      [--crop --out-width 400 --out-height 400] \
      [--equalize-samples] \
      [--debug]

Arguments:
    --data-path                   Base input directory (default: ".")
    --out-path                    Output directory for all generated files
    --start                       First pipeline chapter to run (0–5, default: 0)
    --end                         Last pipeline chapter to run (0–5, default: 6)
    --debug                       Enable DEBUG logging

    # Video Conversion / Frame Counting
    --optimize-counting           Fast .mp4 frame counting (uses metadata)
    --max-workers-frame-counter   Workers for frame counting (default: 20)

    # Background Subtraction
    --background-subtraction-type MOG2|KNN (default: None)
    --max-workers-background-subtraction Workers for parallel subtraction (default: 10)

    # Dataset Creation
    --fps                         Frames per second (default: 25)
    --starting-frame              Start frame offset (default: 1)
    --frame-interval              Frame interval between segments (default: 0)
    --end-frame-buffer            Frames to buffer at end (default: 0)
    --test-by-time                Time‑based dataset splitting
    --time-splits                 Number of time splits (default: 3)
    --files                       Comma‑separated log files list
    --each-video-one-class        Treat each video as its own class

    # Data Splitting
    --k                           Number of folds for cross‑validation (default: 3)
    --model                       Model name for training scripts (default: alexnet)
    --seed                        RNG seed for splitting (default: "01011970")
    --only_split                  Exit after CSV splitting
    --training_only               Generate only training scripts
    --crop_x_offset, --crop_y_offset  Crop offsets (px, default: 0)
    --width, --height             Image dimensions for splitting/training (default: 960×720)

    # Video Sampling
    --number-of-samples           Max samples per video (default: 40000)
    --max-workers-video-sampling  Workers for sampling (default: 3)
    --frames-per-sample           Frames per sample (default: 1)
    --normalize                   Normalize images (default: True)
    --out-channels                Output channels (default: 1)
    --crop                        Enable cropping to output size
    --out-width, --out-height     Crop dimensions (px, default: 400×400)
    --equalize-samples            Equalize class sample counts
    --dataset-writing-batch-size  Batch size for tar writing (default: 30)
    --max-threads-pic-saving      Threads for image saving (default: 4)
    --max-batch-size-sampling     Sample batch size (default: 5)
    --max-workers-tar-writing     Workers for tar writing (default: 4)

    # Training
    --epochs                      Number of training epochs (default: 10)
    --gpus                        GPUs per SLURM job (default: 1)
    --binary-training-optimization
                                  Convert and train with binary files
    --use-dataloader-workers      Enable DataLoader multiprocessing
    --max-dataloader-workers      Number of DataLoader workers (default: 3)
    --loss-fn                     Loss function choice (default: CrossEntropyLoss)
    --gradcam-cnn-model-layer     GradCAM layers list (default: model_a.4.0, model_b.4.0)
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
        "--in-path",
        type=str,
        help='(unifier)path to the data and other input files, default "."',
        default=".",
        required=False,
    )
    parser.add_argument(
        "--out-path",
        type=str,
        help=
        "(unifier)the path where the output is going to appear, default '.'",
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


if __name__ == "__main__":
    get_args()
