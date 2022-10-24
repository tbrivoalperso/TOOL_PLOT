#!/bin/sh
#SBATCH -J PLOTS
#SBATCH -N 1
#SBATCH -p normal256
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=06:00:00
module load cdo/1.9.3

# Folders (plots will make VER2 - VER1)
export FLDR_VERSION1=/scratch/work/brivoalt/RUNS_NEMO/eNEATL36_AGRIF_vvl/EXP01_AGRIF_emodNET_finaldomain_topped_update_test_smooth2
export FLDR_VERSION2=/scratch/work/brivoalt/RUNS_NEMO/eNEATL36_AGRIF_vvl/EXP01_AGRIF_emodNET_finaldomain_from_parent4
export CFG_NAME="eNEATL36"
export AGRIF_CFG_NAME="1_AGRIF"
export VER1="AGRIF"
export VER2="AGRIF_FM_PARENT"
export IS_AGRIF_VER1=true
export IS_AGRIF_VER2=true
export DSTART="2017-09-01"
export DEND="2017-09-12"

export #ZOOM DEFINITION
export IMIN=312
export IMAX=752
export JMIN=751
export JMAX=1176

# SCRIPTS DIRECTORIES
export OUT_DIR=/scratch/work/brivoalt/PLOTS/PLOTS_AGRIF/TEST_PLOTS # Where to plot
export PLOTS_DIR=/scratch/work/brivoalt/PLOTS/TOOL_PLOT/SRC_PLOTS # SRC PLOT FILES
export SCRIPT_DIR=/scratch/work/brivoalt/PLOTS/TOOL_PLOT/SRC # SRC rebuild & Pre-processing files

# LISTS OF PLOTS 
# On mother domain
scripts_motherdomain="MAP_SST.py MAP_KE.py MAP_MLD.py MAP_SSS.py" 
scripts_motherdomain_cmp="MAP_SST_CMP.py MAP_KE_CMP.py MAP_MLD_CMP.py MAP_SSS_CMP.py"

# On agrif
scripts_agrif="" #  MAP_KE.py MAP_MLD.py MAP_SSS.py"

# WHAT SHOULD BE DONE
export DO_PLOTS_MOTHER=false 
export DO_PLOTS_ZOOM=true
export DO_CMP=false # Compare 2 runs or not 
export DO_CMP_ZOOM=true # Compare 2 zooms or not 

export PREPROCESSING=false
export REBUILD=false
export CLEAN_AFTER_REBUILD=false
# Inputs
export SST_OSTIA=/scratch/work/brivoalt/DATA/SST/OSTIA_SST_ORIGIN_y2017-2018_eNEATL36.nc

# Grid names
export gridT_1d_name="gridT"
export gridU_1d_name="gridU"
export gridV_1d_name="gridV"
export gridS_1d_name="gridS"
export grid2D_1d_name="grid2D"

mkdir $OUT_DIR
cd $OUT_DIR
################# DATA REBUILD  ###########################
if $REBUILD;
then
    for FLDR in FLDR_VERSION1 #$FLDR_VERSION1 $FLDR_VERSION2
    do
        cd $FLDR
        cp ${SCRIPT_DIR}/rebuild_and_clean_zoom.sh .
        cp ${SCRIPT_DIR}/rebuild_and_clean_mother.sh .
        sbatch rebuild_and_clean_zoom.sh 
        sbatch rebuild_and_clean_mother.sh 
    done
fi
############ DATA PREPROCESSING ###########################
if $PREPROCESSING ; 
then
    cp ${SCRIPT_DIR}/merge_files.sh .
    sbatch merge_files.sh $FLDR_VERSION2 $IS_AGRIF_VER2
    sbatch merge_files.sh $FLDR_VERSION1 $IS_AGRIF_VER1   
fi

################## PLOTTING #######################
ind=0
VER_LIST=($VER1 $VER2)
if $DO_PLOTS_MOTHER;
then
    cd $OUT_DIR
    for FLDR in $FLDR_VERSION1 $FLDR_VERSION2
    do
    export VER=${VER_LIST[${ind}]}
    echo $VER 
       for script in $scripts_motherdomain ;
       do
          ${SCRIPT_DIR}/sed_and_launch_plots.sh $script $CFG_NAME false false $FLDR     
           
       done
    ind=$(( $ind + 1 ))   
    done
    
    if $DO_CMP ;
    then
        for script in $scripts_motherdomain_cmp ;
        do
           ${SCRIPT_DIR}/sed_and_launch_plots.sh $script $CFG_NAME false true $FLDR_VERSION1 $FLDR_VERSION2     

        done
    fi
fi

ind=0
IS_AGRIF_LIST=($IS_AGRIF_VER1 $IS_AGRIF_VER2)
if $DO_PLOTS_ZOOM;
then
    cd $OUT_DIR
    for FLDR in $FLDR_VERSION1 $FLDR_VERSION2
    do
    if ${IS_AGRIF_LIST[${ind}]} ;
    then
    export VER=${VER_LIST[${ind}]}
    echo $VER 
       for script in $scripts_motherdomain ;
       do
          ${SCRIPT_DIR}/sed_and_launch_plots.sh $script $AGRIF_CFG_NAME true false $FLDR

       done
    ind=$(( $ind + 1 ))
    fi
    done

    if $DO_CMP_ZOOM ;
    then
        for script in $scripts_motherdomain_cmp ;
        do
           ${SCRIPT_DIR}/sed_and_launch_plots.sh $script $AGRIF_CFG_NAME true true $FLDR_VERSION1 $FLDR_VERSION2

        done
    fi
fi


