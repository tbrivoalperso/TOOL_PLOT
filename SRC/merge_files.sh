#!/bin/sh
#SBATCH -J MERGE
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --time=06:00:00
#SBATCH --account=cmems
module load cdo/1.9.3
SST_AVHRR='/scratch/work/legalloudeco/PERMANENT_SPACE/OBSERVATIONS/SST/AVHRR/ORIGIN_GRID/dt/avhrr-only-v2.20170101.nc'
############ DATA PREPROCESSING ###########################
FLDR=$1
IS_AGRIF=$2
cd $FLDR
if $IS_AGRIF
then
    echo "PREPROCESSING AGRIF FILES"
    cdo select,levidx=1 ${AGRIF_CFG_NAME}_1d_${gridT_1d_name}_????????-????????.nc ${AGRIF_CFG_NAME}_1d_gridT.nc
    cdo select,levidx=1 ${AGRIF_CFG_NAME}_1d_${gridU_1d_name}_????????-????????.nc ${AGRIF_CFG_NAME}_1d_gridU.nc
    cdo select,levidx=1 ${AGRIF_CFG_NAME}_1d_${gridV_1d_name}_????????-????????.nc ${AGRIF_CFG_NAME}_1d_gridV.nc
    cdo select,levidx=1 ${AGRIF_CFG_NAME}_1d_${gridS_1d_name}_????????-????????.nc ${AGRIF_CFG_NAME}_1d_gridS.nc
   cdo select,levidx=1 ${AGRIF_CFG_NAME}_1d_${grid2D_1d_name}_????????-????????.nc ${AGRIF_CFG_NAME}_1d_grid2D.nc
   cdo select,levidx=1 ${AGRIF_CFG_NAME}_1d_${grid2D_ssh_1d_name}_????????-????????.nc ${AGRIF_CFG_NAME}_1d_grid2D_ssh.nc

    ncks -v vozocrtx ${AGRIF_CFG_NAME}_1d_gridU.nc ${AGRIF_CFG_NAME}_1d_gridU_tmp.nc
    cdo mul ${AGRIF_CFG_NAME}_1d_gridU_tmp.nc ${AGRIF_CFG_NAME}_1d_gridU_tmp.nc ${AGRIF_CFG_NAME}_1d_gridU2.nc
    rm -f ${AGRIF_CFG_NAME}_1d_gridU_tmp.nc
    ncrename -v vozocrtx,KE2 ${AGRIF_CFG_NAME}_1d_gridU2.nc ${AGRIF_CFG_NAME}_1d_gridU2_tmp.nc
    mv ${AGRIF_CFG_NAME}_1d_gridU2_tmp.nc ${AGRIF_CFG_NAME}_1d_gridU2.nc
    ncks -v vomecrty ${AGRIF_CFG_NAME}_1d_gridV.nc ${AGRIF_CFG_NAME}_1d_gridV_tmp.nc
    cdo mul ${AGRIF_CFG_NAME}_1d_gridV_tmp.nc ${AGRIF_CFG_NAME}_1d_gridV_tmp.nc ${AGRIF_CFG_NAME}_1d_gridV2.nc
    rm -f ${AGRIF_CFG_NAME}_1d_gridV_tmp.nc
    ncrename -v vomecrty,KE2 ${AGRIF_CFG_NAME}_1d_gridV2.nc ${AGRIF_CFG_NAME}_1d_gridV2_tmp.nc
    mv ${AGRIF_CFG_NAME}_1d_gridV2_tmp.nc ${AGRIF_CFG_NAME}_1d_gridV2.nc
    cdo add ${AGRIF_CFG_NAME}_1d_gridU2.nc ${AGRIF_CFG_NAME}_1d_gridV2.nc ${AGRIF_CFG_NAME}_1d_KE2.nc
    rm ${AGRIF_CFG_NAME}_1d_gridU2.nc ${AGRIF_CFG_NAME}_1d_gridV2.nc
    cdo monmean ${AGRIF_CFG_NAME}_1d_KE2.nc ${AGRIF_CFG_NAME}_1m_KE2.nc                                 
    cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36_zoom.nc ${AGRIF_CFG_NAME}_1d_gridT.nc ${AGRIF_CFG_NAME}_1d_gridT.nc_tmp
    mv ${AGRIF_CFG_NAME}_1d_gridT.nc_tmp ${AGRIF_CFG_NAME}_1d_gridT.nc
    cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36_zoom.nc ${AGRIF_CFG_NAME}_1d_gridS.nc ${AGRIF_CFG_NAME}_1d_gridS.nc_tmp
    mv ${AGRIF_CFG_NAME}_1d_gridS.nc_tmp ${AGRIF_CFG_NAME}_1d_gridS.nc
    cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36_zoom.nc ${AGRIF_CFG_NAME}_1d_gridU.nc ${AGRIF_CFG_NAME}_1d_gridU.nc_tmp
    mv ${AGRIF_CFG_NAME}_1d_gridU.nc_tmp ${AGRIF_CFG_NAME}_1d_gridU.nc
    cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36_zoom.nc ${AGRIF_CFG_NAME}_1d_gridV.nc ${AGRIF_CFG_NAME}_1d_gridV.nc_tmp
    mv ${AGRIF_CFG_NAME}_1d_gridV.nc_tmp ${AGRIF_CFG_NAME}_1d_gridV.nc
    cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36_zoom.nc ${AGRIF_CFG_NAME}_1d_KE2.nc ${AGRIF_CFG_NAME}_1d_KE2.nc_tmp
    mv ${AGRIF_CFG_NAME}_1d_KE2.nc_tmp ${AGRIF_CFG_NAME}_1d_KE2.nc
    cdo remapbil,${SST_OSTIA} ${AGRIF_CFG_NAME}_1m_gridT.nc ${AGRIF_CFG_NAME}_1m_gridT_OSTIA.nc
    cdo remapbil,${SST_OSTIA} ${AGRIF_CFG_NAME}_1d_gridT.nc ${AGRIF_CFG_NAME}_1d_gridT_OSTIA.nc
    cdo remapbil,${SST_AVHRR} ${AGRIF_CFG_NAME}_1d_gridT.nc ${AGRIF_CFG_NAME}_1d_gridT_AVHRR.nc
else

cdo select,levidx=1 ${CFG_NAME}_1d_${gridT_1d_name}_????????-????????.nc ${CFG_NAME}_1d_gridT.nc
cdo select,levidx=1 ${CFG_NAME}_1d_${gridU_1d_name}_????????-????????.nc ${CFG_NAME}_1d_gridU.nc
cdo select,levidx=1 ${CFG_NAME}_1d_${gridV_1d_name}_????????-????????.nc ${CFG_NAME}_1d_gridV.nc
cdo select,levidx=1 ${CFG_NAME}_1d_${gridS_1d_name}_????????-????????.nc ${CFG_NAME}_1d_gridS.nc
cdo select,levidx=1 ${CFG_NAME}_1d_${grid2D_1d_name}_????????-????????.nc ${CFG_NAME}_1d_grid2D.nc
cdo select,levidx=1 ${CFG_NAME}_1d_${grid2D_ssh_1d_name}_????????-????????.nc ${CFG_NAME}_1d_grid2D_ssh.nc

#
cdo monmean ${CFG_NAME}_1d_gridT.nc ${CFG_NAME}_1m_gridT.nc
cdo monmean ${CFG_NAME}_1d_gridU.nc ${CFG_NAME}_1m_gridU.nc
cdo monmean ${CFG_NAME}_1d_gridV.nc ${CFG_NAME}_1m_gridV.nc
cdo monmean ${CFG_NAME}_1d_gridS.nc ${CFG_NAME}_1m_gridS.nc
cdo monmean ${CFG_NAME}_1d_grid2D.nc ${CFG_NAME}_1m_grid2D.nc

ncks -v vozocrtx ${CFG_NAME}_1d_gridU.nc ${CFG_NAME}_1d_gridU_tmp.nc
cdo mul ${CFG_NAME}_1d_gridU_tmp.nc ${CFG_NAME}_1d_gridU_tmp.nc ${CFG_NAME}_1d_gridU2.nc
rm -f ${CFG_NAME}_1d_gridU_tmp.nc
ncrename -v vozocrtx,KE2 ${CFG_NAME}_1d_gridU2.nc ${CFG_NAME}_1d_gridU2_tmp.nc
mv ${CFG_NAME}_1d_gridU2_tmp.nc ${CFG_NAME}_1d_gridU2.nc
ncks -v vomecrty ${CFG_NAME}_1d_gridV.nc ${CFG_NAME}_1d_gridV_tmp.nc
cdo mul ${CFG_NAME}_1d_gridV_tmp.nc ${CFG_NAME}_1d_gridV_tmp.nc ${CFG_NAME}_1d_gridV2.nc
rm -f ${CFG_NAME}_1d_gridV_tmp.nc
ncrename -v vomecrty,KE2 ${CFG_NAME}_1d_gridV2.nc ${CFG_NAME}_1d_gridV2_tmp.nc
mv ${CFG_NAME}_1d_gridV2_tmp.nc ${CFG_NAME}_1d_gridV2.nc
cdo add ${CFG_NAME}_1d_gridU2.nc ${CFG_NAME}_1d_gridV2.nc ${CFG_NAME}_1d_KE2.nc
rm ${CFG_NAME}_1d_gridU2.nc ${CFG_NAME}_1d_gridV2.nc
cdo monmean ${CFG_NAME}_1d_KE2.nc ${CFG_NAME}_1m_KE2.nc


cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36.nc ${CFG_NAME}_1d_gridT.nc ${CFG_NAME}_1d_gridT.nc_tmp
mv ${CFG_NAME}_1d_gridT.nc_tmp ${CFG_NAME}_1d_gridT.nc
cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36.nc ${CFG_NAME}_1d_gridS.nc ${CFG_NAME}_1d_gridS.nc_tmp
mv ${CFG_NAME}_1d_gridS.nc_tmp ${CFG_NAME}_1d_gridS.nc
cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36.nc ${CFG_NAME}_1d_gridU.nc ${CFG_NAME}_1d_gridU.nc_tmp
mv ${CFG_NAME}_1d_gridU.nc_tmp ${CFG_NAME}_1d_gridU.nc
cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36.nc ${CFG_NAME}_1d_gridV.nc ${CFG_NAME}_1d_gridV.nc_tmp
mv ${CFG_NAME}_1d_gridV.nc_tmp ${CFG_NAME}_1d_gridV.nc
cdo setgrid,/scratch/work/brivoalt/RUNS_NEMO/grid_eNEATL36.nc ${CFG_NAME}_1d_KE2.nc ${CFG_NAME}_1d_KE2.nc_tmp
mv ${CFG_NAME}_1d_KE2.nc_tmp ${CFG_NAME}_1d_KE2.nc

# Remap SST on OSTIA
cdo remapbil,${SST_OSTIA} ${CFG_NAME}_1m_gridT.nc ${CFG_NAME}_1m_gridT_OSTIA.nc
cdo remapbil,${SST_OSTIA} ${CFG_NAME}_1d_gridT.nc ${CFG_NAME}_1d_gridT_OSTIA.nc
cdo remapbil,${SST_AVHRR} ${CFG_NAME}_1d_gridT.nc ${CFG_NAME}_1d_gridT_AVHRR.nc

fi
