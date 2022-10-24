#!/bin/sh
#SBATCH -J FLT
#SBATCH -N 1
#SBATCH -p normal256
#SBATCH --no-requeue
#SBATCH --time=00:30:00
##SBATCH --account=cmems
module load cdo/1.9.3
ulimit -s unlimited


index=$1
script=$2  # SCRIPT FILE
CFG_NM=$3
AGRIF=$4
DO_CMP=$5    # PLOT ONLY FLDR
FLDR_1=$6
FLDR_2=$7

echo $script
if $DO_CMP;
then
    echo "DO_COMP"
    sed -e "s%FLDR_VERSION1%`echo $FLDR_1`%g" \
        -e "s%FLDR_VERSION2%`echo $FLDR_2`%g" \
        -e "s%CFG_NM%`echo $CFG_NM`%g" \
        -e "s%AGRIF_CFG_NAME%`echo $AGRIF_CFG_NAME`%g" \
        -e "s%CFG_NAME%`echo $CFG_NAME`%g" \
        -e "s%SEASON%`echo $SEASON`%g" \
        -e "s%VER1%`echo $VER1`%g" \
        -e "s%VER2%`echo $VER2`%g" \
        -e "s%MONTH_ana%`echo $MONTH_ana`%g" \
        -e "s%DSTART_TS%`echo $DSTART_TS`%g" \
        -e "s%DSTART%`echo $DSTART`%g" \
        -e "s%IMIN%`echo $IMIN`%g" \
        -e "s%IMAX%`echo $IMAX`%g" \
        -e "s%JMIN%`echo $JMIN`%g" \
        -e "s%JMAX%`echo $JMAX`%g" \
        -e "s%DEND%`echo $DEND`%g" ${PLOTS_DIR}/${script} > ${script}.${index}

else
    sed -e "s%FLDR%`echo $FLDR_1`%g" \
        -e "s%AGRIF_CFG_NAME%`echo $AGRIF_CFG_NAME`%g" \
        -e "s%CFG_NAME%`echo $CFG_NAME`%g" \
        -e "s%CFG_NM%`echo $CFG_NM`%g" \
        -e "s%SEASON%`echo $SEASON`%g" \
        -e "s%VER%`echo $VER`%g" \
        -e "s%DSTART%`echo $DSTART`%g" \
        -e "s%MONTH_ana%`echo $MONTH_ana`%g" \
        -e "s%IMIN%`echo $IMIN`%g" \
        -e "s%IMAX%`echo $IMAX`%g" \
        -e "s%JMIN%`echo $JMIN`%g" \
        -e "s%JMAX%`echo $JMAX`%g" \
        -e "s%DEND%`echo $DEND`%g" ${PLOTS_DIR}/${script} > ${script}.${index}
fi

if $AGRIF
  then
         sed -i -e "s%domain_cfg.nc%1_domain_cfg.nc%g" ${script}.${index}
         sed -i -e "s%res_harm_ssh.nc%1_res_harm_ssh.nc%g" ${script}.${index}
         sed -i -e "s%AGRIF_bool%False%g" ${script}.${index}
         sed -i -e "s%CFGNM2%${CFG_NAME}%g" ${script}.${index}
  else
         sed -i -e "s%AGRIF_bool%True%g" ${script}.${index}
fi
if $IS_AGRIF_VER1; then sed -i -e "s%ISAGRIF1=False%ISAGRIF1=True%g" ${script}.${index} ; fi
if $IS_AGRIF_VER2; then sed -i -e "s%ISAGRIF2=False%ISAGRIF2=True%g" ${script}.${index} ; fi
chmod +x ${script}.${index}
srun python ${script}.${index}
