#!/bin/sh

module load cdo/1.9.3

# MAKE PLOTS ?
export DO_PLOTS_MOTHER=true 
export DO_PLOTS_ZOOM=false
export DO_GIFS_MOTHER=false # Make gif movies
export DO_GIFS_ZOOM=false # Make gif movies

export DO_CMP=true # Compare 2 runs or not 
export DO_CMP_ZOOM=false # Compare 2 zooms or not 
# MAKE PREPROCESSING ?
export PREPROCESSING=false
export PREPROCESS_NOOBS=false
# MAKE REBUILD ?
export REBUILD=false
export CLEAN_AFTER_REBUILD=false

# OTHER VARIOUS COMPUTATIONS ?
export DO_CURL_MEAN=false
export DO_CURL_ALL=false
export DO_HARM=false

# COMPUTE NOOBS OLAFILES
export NOOBS=false
export COMBINE_NOOBS=false
export DO_PLOT_NOOBS=false
export FILTER=false
# ----------------------------------------------------------
# DO NOT CHANGE
WORK_DIR=$(pwd)
PARAMETERS=$1
ENV_PARAMETERS=${WORK_DIR}/${PARAMETERS}
source $ENV_PARAMETERS 
echo $SCRIPT_DIR

#sbatch ${SCRIPT_DIR}/MAIN.sh
${SCRIPT_DIR}/MAIN.sh
