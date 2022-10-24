#!/bin/sh
#SBATCH -J CURL
#SBATCH -N 1
#SBATCH -p normal256
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=06:00:00
module load cdo/1.9.3
ulimit -s unlimited
############ DATA PREPROCESSING ###########################
FLDR=$1
IS_AGRIF=$2
input_start=$DSTART_TS
input_end=$DEND

startdate=$(date -I -d "$input_start") || exit -1
enddate=$(date -I -d "$input_end")     || exit -1

d="$startdate"
datefmt=$(date -d $d +%Y%m%d)
while [ "$d" != "$enddate" ]; do
  echo $datefmt
  datefmt=$(date -d $d +%Y%m%d)
  d=$(date -I -d "$d + 1 day")
  
  cd $FLDR
  pwd 
  if $IS_AGRIF
  then
     ln -fs $MESH_MASK_ZOOM mesh_hgr.nc
     ln -fs $MESH_MASK_ZOOM mesh_zgr.nc
     ln -fs ${CDFTOOLS}/cdfcurl .
     ./cdfcurl -u ${AGRIF_CFG_NAME}_1h_${gridU_1h_name}_${datefmt}-${datefmt}.nc sozocrtx -v ${AGRIF_CFG_NAME}_1h_${gridV_1h_name}_${datefmt}-${datefmt}.nc somecrty -surf  
     mv curl.nc ${AGRIF_CFG_NAME}_1h_curl_${datefmt}-${datefmt}.nc
  
     unlink mesh_hgr.nc
     unlink mesh_zgr.nc
  fi
  
  ln -fs $MESH_MASK_MOTHER mesh_hgr.nc
  ln -fs $MESH_MASK_MOTHER mesh_zgr.nc
  ln -fs ${CDFTOOLS}/cdfcurl .
  ./cdfcurl -u ${CFG_NAME}_1h_${gridU_1h_name}_${datefmt}-${datefmt}.nc sozocrtx -v ${CFG_NAME}_1h_${gridV_1h_name}_${datefmt}-${datefmt}.nc somecrty -surf
  mv curl.nc ${CFG_NAME}_1h_curl_${datefmt}-${datefmt}.nc 
  unlink mesh_hgr.nc
  unlink mesh_zgr.nc
done
