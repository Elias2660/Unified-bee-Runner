#!/bin/bash
source ../venv/bin/activate
cd bee_analysis
echo "Submitting Slurm job for saliency map generation..."

# Submit the job with high resource allocation
sbatch \
  --job-name=saliency_generation \
  -n 2 \
  -c 4 \
  -G 2 \
  --mem=200G \
  --time=28800 \
  -o "dev/null"\
  --wrap="
    echo 'Job started at: \$(date)'
    echo 'Working directory: \$(pwd)'
    echo 'Python path:'
    python -c 'import sys; print(\"\n\".join(sys.path))'
    
    # Run the visualization generation with high resource settings
    python3 generate_visualizations.py \
        --checkpoint ../../model.checkpoint \
        --datasets ../../dataset_0.tar ../../dataset_1.tar ../../dataset_2.tar \
        --no_gradcam \
        --debug
        
    echo 'Job completed at: \$(date)'
  "

echo "Slurm submitted."