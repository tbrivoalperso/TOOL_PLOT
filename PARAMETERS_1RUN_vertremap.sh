#!/bin/sh
#SBATCH -J PLOTS
#SBATCH -N 1
#SBATCH -p normal256
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=06:00:00
module load cdo/1.9.3

# Folders (plots will make VER2 - VER1)
export FLDR_VERSION1=/scratch/work/brivoalt/RUNS_NEMO/eNEATL36_AGRIF_vvl/EXP01_AGRIF_emodNET_finaldomain_vertremap
export FLDR_VERSION2=
export CFG_NAME="eNEATL36"
export AGRIF_CFG_NAME="1_AGRIF"
export VER1="AGRIF"
export VER2=
export IS_AGRIF_VER1=true
export IS_AGRIF_VER2=true # cannot be true if IS_AGRIF_VER1=false
export DSTART="2017-09-01" # Start date for maps
export DSTART_TS="2017-08-23" # Start date for time series and gifs (usually the beginning of the run
export DEND="2017-09-15"

export #ZOOM DEFINITION
export IMIN=220
export IMAX=1073
export JMIN=350
export JMAX=1245

# SCRIPTS DIRECTORIES
export OUT_DIR=/scratch/work/brivoalt/PLOTS/PLOTS_AGRIF/AGRIF_VERTREMAP2 # Where to plot
export PLOTS_DIR=/scratch/work/brivoalt/PLOTS/TOOL_PLOT/SRC_PLOTS # SRC PLOT FILES
export SCRIPT_DIR=/scratch/work/brivoalt/PLOTS/TOOL_PLOT/SRC # SRC rebuild & Pre-processing files
export CDFTOOLS=/home/ext/mr/smer/samsong/CDFTOOLS/bin/
# LISTS OF PLOTS 
# On mother domain
export scripts_motherdomain="" #"MAP_tide.py MAP_VOR.py MAP_SST.py MAP_KE.py MAP_MLD.py MAP_SSS.py" 
export scripts_motherdomain_cmp="TS_SSS_CMP.py TS_MLD_CMP.py TS_SST_CMP.py TS_KE_CMP.py" #"MAP_SST_CMP.py MAP_KE_CMP.py MAP_MLD_CMP.py MAP_SSS_CMP.py"

 #On agrif
export scripts_zoom="" #"MAP_tide.py MAP_VOR.py MAP_SST.py MAP_KE.py MAP_MLD.py MAP_SSS.py" 
export scripts_zoom_cmp="TS_SSS_CMP.py TS_MLD_CMP.py TS_SST_CMP.py TS_KE_CMP.py" #"MAP_SST_CMP.py MAP_KE_CMP.py MAP_MLD_CMP.py MAP_SSS_CMP.py"

# Scripts to make gifs
export scripts_gifs="MAP_SST_GIFS.py" #MAP_KE_GIFS_manche.py " #MAP_VOR_GIFS_manche.py" #"MAP_SST_GIFS_manche.py" #"MAP_SST_GIFS.py MAP_KE_GIFS.py MAP_VOR_GIFS.py" 
export scripts_gifs_daily="" #"MAP_VOR_GIFS_daily.py"

# Inputs
export SST_OSTIA=/scratch/work/brivoalt/DATA/SST/OSTIA_SST_ORIGIN_y2017-2018_eNEATL36.nc
export MESH_MASK_MOTHER=/scratch/work/brivoalt/DATA_eNEATL36/mesh_mask_eNEATL36.nc
export MESH_MASK_ZOOM=/scratch/work/brivoalt/DATA_eNEATL36/mesh_mask_eNEATL36_zoom.nc

# Grid names
export gridT_1d_name="gridT"
export gridT_1h_name="gridT"
export grid2D_1d_name="grid2D"
export grid2D_1h_name="gridT"
export gridU_1d_name="gridU"
export gridU_1h_name="gridU"
export gridV_1d_name="gridV"
export gridV_1h_name="gridV"
export gridS_1d_name="gridS"



