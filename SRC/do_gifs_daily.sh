#!/bin/sh
#SBATCH -J PLOTS
#SBATCH -N 1
#SBATCH -p normal256
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=06:00:00
#SBATCH --account=cmems
module load cdo/1.9.3

script=$1  # SCRIPT FILE
CFG_NM=$2
AGRIF=$3
DO_CMP=$4    # PLOT ONLY FLDR
FLDR_1=$5
FLDR_2=$6

input_start=$DSTART_TS
input_end=$DEND
SCRIPTID=${script%.py}
if $AGRIF
then
  script_new=${script}.AGRIF
else
  script_new=${script}.MOTHER
fi
# After this, startdate and enddate will be valid ISO 8601 dates,
# or the script will have aborted when it encountered unparseable data
# such as input_end=abcd

startdate=$(date -I -d "$input_start") || exit -1
enddate=$(date -I -d "$input_end")     || exit -1

d="$startdate"
datefmt=$(date -d $d +%Y%m%d)
while [ "$d" != "$enddate" ]; do 
     echo $datefmt
     datefmt=$(date -d $d +%Y%m%d)
     d=$(date -I -d "$d + 1 day")
     if $AGRIF
     then
       TFILENM=${AGRIF_CFG_NAME}_1d_${gridT_1d_name}_${datefmt}-${datefmt}.nc
       UFILENM=${AGRIF_CFG_NAME}_1d_${gridU_1d_name}_${datefmt}-${datefmt}.nc
       VFILENM=${AGRIF_CFG_NAME}_1d_${gridV_1d_name}_${datefmt}-${datefmt}.nc
       FFILENM=${AGRIF_CFG_NAME}_1d_curl_${datefmt}-${datefmt}.nc
   
     else
       TFILENM=${CFG_NAME}_1d_${gridT_1d_name}_${datefmt}-${datefmt}.nc
       UFILENM=${CFG_NAME}_1d_${gridU_1d_name}_${datefmt}-${datefmt}.nc
       VFILENM=${CFG_NAME}_1d_${gridV_1d_name}_${datefmt}-${datefmt}.nc
       FFILENM=${CFG_NAME}_1d_curl_${datefmt}-${datefmt}.nc
       FSONFILENM=${AGRIF_CFG_NAME}_1d_curl_${datefmt}-${datefmt}.nc
 
     fi
   
   echo $script
   if $DO_CMP;
   then
       echo "DO_COMP"
       sed -e "s%FLDR_VERSION1%`echo $FLDR_1`%g" \
           -e "s%FLDR_VERSION2%`echo $FLDR_2`%g" \
           -e "s%UFILENM%`echo $UFILENM`%g" \
           -e "s%VFILENM%`echo $VFILENM`%g" \
           -e "s%TFILENM%`echo $TFILENM`%g" \
           -e "s%FFILENM%`echo $FFILENM`%g" \
           -e "s%FSONFILENM%`echo $FSONFILENM`%g" \
           -e "s%CFG_NM%`echo $CFG_NM`%g" \
           -e "s%SEASON%`echo $SEASON`%g" \
           -e "s%VER1%`echo $VER1`%g" \
           -e "s%VER2%`echo $VER2`%g" \
           -e "s%DSTART_TS%`echo $DSTART_TS`%g" \
           -e "s%DSTART%`echo $DSTART`%g" \
           -e "s%IMIN%`echo $IMIN`%g" \
           -e "s%IMAX%`echo $IMAX`%g" \
           -e "s%JMIN%`echo $JMIN`%g" \
           -e "s%JMAX%`echo $JMAX`%g" \
           -e "s%SCRIPTID%`echo $SCRIPTID`%g" \
           -e "s%DEND%`echo $DEND`%g" ${PLOTS_DIR}/DO_GIFS/${script} > ${script_new}
   else
       sed -e "s%FLDR%`echo $FLDR_1`%g" \
           -e "s%UFILENM%`echo $UFILENM`%g" \
           -e "s%VFILENM%`echo $VFILENM`%g" \
           -e "s%TFILENM%`echo $TFILENM`%g" \
           -e "s%FFILENM%`echo $FFILENM`%g" \
           -e "s%FSONFILENM%`echo $FSONFILENM`%g" \
           -e "s%CFG_NM%`echo $CFG_NM`%g" \
           -e "s%SEASON%`echo $SEASON`%g" \
           -e "s%VER%`echo $VER`%g" \
           -e "s%DSTART%`echo $DSTART`%g" \
           -e "s%IMIN%`echo $IMIN`%g" \
           -e "s%IMAX%`echo $IMAX`%g" \
           -e "s%JMIN%`echo $JMIN`%g" \
           -e "s%JMAX%`echo $JMAX`%g" \
           -e "s%SCRIPTID%`echo $SCRIPTID`%g" \
           -e "s%DEND%`echo $DEND`%g" ${PLOTS_DIR}/DO_GIFS/${script} > ${script_new}
   fi
   
   if $AGRIF
   then
          sed -i -e "s%domain_cfg.nc%1_domain_cfg.nc%g" ${script_new}
          sed -i -e "s%res_harm_ssh.nc%1_res_harm_ssh.nc%g" ${script_new}
          sed -i -e "s%AGRIF_bool%False%g" ${script_new}
   else
          sed -i -e "s%AGRIF_bool%True%g" ${script_new}
   fi
   
   python ${script_new}
done

if $DO_CMP
then
  echo "done"
else
  echo "process gif on files :" ${SCRIPTID}_${CFG_NM}_1d_${VER}_*.png
  convert -delay 30 -loop 0 ${SCRIPTID}_${CFG_NM}_1d_${VER}_*.png ${SCRIPTID}_${CFG_NM}_1d_${VER}.gif
  #convert -delay 30 -loop 0 -quality 20 ${SCRIPTID}_${CFG_NM}_1d_${VER}_*.png ${SCRIPTID}_${CFG_NM}_1d_${VER}_q20.gif
  mv ${SCRIPTID}_${CFG_NM}_1d_${VER}.gif ${OUT_DIR}/
fi
