#!/bin/sh
#SBATCH -J CURL
#SBATCH -N 1
#SBATCH -p normal256
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=06:00:00
#SBATCH --account=cmems
module load cdo/1.9.3
ulimit -s unlimited
############ DATA PREPROCESSING ###########################
FLDR=$1
IS_AGRIF=$2

cd $FLDR

if $IS_AGRIF
then
   ln -fs $MESH_MASK_ZOOM mesh_hgr.nc
   ln -fs $MESH_MASK_ZOOM mesh_zgr.nc
   ln -fs ${CDFTOOLS}/cdfcurl .
   ./cdfcurl -u ${AGRIF_CFG_NAME}_1d_gridU.nc vozocrtx -v ${AGRIF_CFG_NAME}_1d_gridV.nc vomecrty -surf  
   mv curl.nc ${AGRIF_CFG_NAME}_1d_curl.nc

   unlink mesh_hgr.nc
   unlink mesh_zgr.nc
fi




ln -fs $MESH_MASK_ZOOM mesh_hgr.nc
ln -fs $MESH_MASK_ZOOM mesh_zgr.nc
ln -fs ${CDFTOOLS}/cdfcurl .
./cdfcurl -u ${CFG_NAME}_1d_gridU.nc vozocrtx -v ${CFG_NAME}_1d_gridV.nc vomecrty -surf       
mv curl.nc ${CFG_NAME}_1d_curl.nc
unlink mesh_hgr.nc
unlink mesh_zgr.nc

