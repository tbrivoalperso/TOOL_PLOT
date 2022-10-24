#!/bin/sh
#SBATCH -J HARM 
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --time=06:00:00
module load cdo/1.9.3
ulimit -s unlimited
NAMELIST_HANA=${SCRIPT_DIR}/HANA/namelist
PROG_HANA=${SCRIPT_DIR}/HANA/tide_ana
############ DATA PREPROCESSING ###########################
FLDR=$1
IS_AGRIF=$2
DSTART=$3
DEND=$4
cd $FLDR
cp $PROG_HANA .
pwd

cdo select,levidx=1 ${CFG_NAME}_1h_${grid2D_1h_name}_????????-????????.nc ${CFG_NAME}_1h_grid2D.nc
cdo seldate,${DSTART},${DEND} ${CFG_NAME}_1h_grid2D.nc ${CFG_NAME}_1h_grid2D_perio.nc
cp $NAMELIST_HANA namelist
./tide_ana ${CFG_NAME}_1h_grid2D_perio.nc
rm ${CFG_NAME}_1h_grid2D_perio.nc


if $IS_AGRIF
then
    echo "PREPROCESSING AGRIF FILES"
    cdo select,levidx=1 ${AGRIF_CFG_NAME}_1h_${grid2D_1h_name}_????????-????????.nc ${AGRIF_CFG_NAME}_1h_grid2D.nc
    cdo seldate,${DSTART},${DEND} ${AGRIF_CFG_NAME}_1h_grid2D.nc ${AGRIF_CFG_NAME}_1h_grid2D_perio.nc
    sed -e "s/res_harm_ssh.nc/`echo 1_res_harm_ssh.nc`/" $NAMELIST_HANA > namelist
    ./tide_ana ${AGRIF_CFG_NAME}_1h_grid2D_perio.nc
    rm ${AGRIF_CFG_NAME}_1h_grid2D_perio.nc
fi

