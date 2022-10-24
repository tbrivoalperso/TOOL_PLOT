#!/bin/bash
#SBATCH -J PLOTS
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH --time=10:10:00
#SBATCH --account=cmems
module load cdo/1.9.3
# ---------------------------------------------------------
#--------------------------------------------------------- 
#                        MAIN PROGRAM
#
# Purpose : 
# - REBUILD FILES
# - MERGE FILES FOR PLOTS
# - PLOT FILES
#

. "/home/ext/mr/smer/brivoalt/anaconda3/etc/profile.d/conda.sh"
conda activate pyenv
ulimit -s unlimited
mkdir $OUT_DIR
cd $OUT_DIR
######################## DATA REBUILD #####################
if $REBUILD;
then
FLDR=$FLDR_VERSION1
cd $FLDR
pwd
cp ${SCRIPT_DIR}/rebuild_and_clean_immerse.sh .
cp ${SCRIPT_DIR}/rebuild_pergrid.sh .
job1=$(sbatch rebuild_and_clean_immerse.sh eNEATL36) 
job2=$(sbatch rebuild_and_clean_immerse.sh 1_AGRIF) 
job1_ID=$(echo $job1 | awk '{print $4}')
job2_ID=$(echo $job2 | awk '{print $4}')

FLDR=$FLDR_VERSION2
cd $FLDR
pwd
cp ${SCRIPT_DIR}/rebuild_and_clean_immerse.sh .
cp ${SCRIPT_DIR}/rebuild_pergrid.sh .

job1_bis=$(sbatch rebuild_and_clean_immerse.sh eNEATL36) 
job2_bis=$(sbatch rebuild_and_clean_immerse.sh 1_AGRIF) 
job1_ID_bis=$(echo $job1 | awk '{print $4}')
job2_ID_bis=$(echo $job2 | awk '{print $4}')


fi
#################### DATA PREPROCESSING ###################
if $PREPROCESSING ; 
then
    cp ${SCRIPT_DIR}/merge_files.sh .
    if $REBUILD ; 
    then
        if $IS_AGRIF_VER1
        then
        job3=$(sbatch --dependency=afterany:$job1_ID:$job2_ID merge_files.sh $FLDR_VERSION1 true )
        job3_ID=$(echo $job3 | awk '{print $4}')
        fi
        job3_bis=$(sbatch --dependency=afterany:$job1_ID:$job2_ID merge_files.sh $FLDR_VERSION1 false )
        job3_bis_ID=$(echo $job3 | awk '{print $4}')

        if $IS_AGRIF_VER2
        then

            job4=$(sbatch --dependency=afterany:$job1_ID_bis:$job2_ID_bis merge_files.sh $FLDR_VERSION2 true )   
            job4_ID=$(echo $job4 | awk '{print $4}')
        fi
        job4_bis=$(sbatch --dependency=afterany:$job1_ID_bis:$job2_ID_bis merge_files.sh $FLDR_VERSION2 false )
        job4_ID_bis=$(echo $job4 | awk '{print $4}')

        echo $job4
    else
        if $IS_AGRIF_VER1
        then
            job3=$(sbatch merge_files.sh $FLDR_VERSION1 true )
            job3_ID=$(echo $job3 | awk '{print $4}')
        fi
        job3_bis=$(sbatch merge_files.sh $FLDR_VERSION1 false )
        job3_bis_ID=$(echo $job3 | awk '{print $4}')

        if $IS_AGRIF_VER2
        then

            job4=$(sbatch merge_files.sh $FLDR_VERSION2 true )          
            job4_ID=$(echo $job4 | awk '{print $4}')
        fi  
        job4_bis=$(sbatch merge_files.sh $FLDR_VERSION2 false )
        job4_ID_bis=$(echo $job4 | awk '{print $4}')

    fi

    if $DO_CURL_MEAN ;
    then
        cp ${SCRIPT_DIR}/compute_vorticity_mean.sh .
        job5=$(sbatch --dependency=afterany:$job3_ID compute_vorticity_mean.sh $FLDR_VERSION1 $IS_AGRIF_VER1 )
        job6=$(sbatch --dependency=afterany:$job4_ID compute_vorticity_mean.sh $FLDR_VERSION2 $IS_AGRIF_VER2 )
    fi
    
else
    #--------------------------------------------------------------
    if $DO_CURL_MEAN ;
    then
        cp ${SCRIPT_DIR}/compute_vorticity_mean.sh .
        job5=$(sbatch compute_vorticity_mean.sh $FLDR_VERSION1 $IS_AGRIF_VER1 )
        job6=$(sbatch compute_vorticity_mean.sh $FLDR_VERSION2 $IS_AGRIF_VER2 )
    fi
fi
if $DO_CURL_ALL
then
    if $REBUILD ;
    then
        cp ${SCRIPT_DIR}/compute_vorticity_all_daily.sh .
        job7=$(sbatch --dependency=afterany:$job1_ID compute_vorticity_all_daily.sh $FLDR_VERSION1 $IS_AGRIF_VER1 )
        job8=$(sbatch --dependency=afterany:$job2_ID compute_vorticity_all_daily.sh $FLDR_VERSION2 $IS_AGRIF_VER2 )

    else
        cp ${SCRIPT_DIR}/compute_vorticity_all_daily.sh .
        job7=$(sbatch compute_vorticity_all_daily.sh $FLDR_VERSION1 $IS_AGRIF_VER1 )
        job8=$(sbatch compute_vorticity_all_daily.sh $FLDR_VERSION2 $IS_AGRIF_VER2 )
    fi
fi


if $DO_HARM
then
    cp ${SCRIPT_DIR}/analyse_harmonique.sh .
    if $REBUILD ;
    then
        job9=$(sbatch --dependency=afterany:$job1_ID:$job2_ID analyse_harmonique.sh $FLDR_VERSION1 $IS_AGRIF_VER1 $DSTART $DEND)
        job10=$(sbatch --dependency=afterany:$job1_ID_bis:$job2_ID_bis analyse_harmonique.sh $FLDR_VERSION2 $IS_AGRIF_VER2 $DSTART $DEND)
    else
        job9=$(sbatch analyse_harmonique.sh $FLDR_VERSION1 $IS_AGRIF_VER1 $DSTART $DEND)
        job10=$(sbatch analyse_harmonique.sh $FLDR_VERSION2 $IS_AGRIF_VER2 $DSTART $DEND)

    fi 
fi
if $PREPROCESS_NOOBS
then

    cp ${SCRIPT_DIR}/LAUNCH_RUN_HARM_AND_NOOBS.sub .
    if $REBUILD ;
    then
        jobnoobs1=$(sbatch --dependency=afterany:$job1_ID:$job2_ID LAUNCH_RUN_HARM_AND_NOOBS.sub $FLDR_VERSION1 $CFG_NAME $DEND)
        jobnoobs2=$(sbatch --dependency=afterany:$job1_ID_bis:$job2_ID_bis LAUNCH_RUN_HARM_AND_NOOBS.sub $FLDR_VERSION2 $CFG_NAME $DEND) 
    else
        jobnoobs1=$(sbatch LAUNCH_RUN_HARM_AND_NOOBS.sub $FLDR_VERSION1 $CFG_NAME $DEND)
        jobnoobs2=$(sbatch LAUNCH_RUN_HARM_AND_NOOBS.sub $FLDR_VERSION2 $CFG_NAME $DEND)
    fi
    jobnoobs1_ID=$(echo $jobnoobs1 | awk '{print $4}')
    jobnoobs2_ID=$(echo $jobnoobs2 | awk '{print $4}')

fi


if $NOOBS
then

    cp ${SCRIPT_DIR}/Launch_noobs.sub .
    if $PREPROCESS_NOOBS
    then
        jobnoobs3=$(sbatch --dependency=afterany:$jobnoobs1_ID Launch_noobs.sub $FLDR_VERSION1 $CYCLE_NOOBS "RUN1")
        jobnoobs4=$(sbatch --dependency=afterany:$jobnoobs2_ID Launch_noobs.sub $FLDR_VERSION2 $CYCLE_NOOBS "RUN2")
    else
        jobnoobs3=$(sbatch Launch_noobs.sub $FLDR_VERSION1 $CYCLE_NOOBS "RUN1")
        jobnoobs4=$(sbatch Launch_noobs.sub $FLDR_VERSION2 $CYCLE_NOOBS "RUN2")
    fi
fi
################## FILTERING #####################
i=1
if $FILTER
then
   for FLDR in $FLDR_VERSION1 $FLDR_VERSION2
   do
    sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh udaily_${i} compute_daily_u_1d.py $CFG_NAME false false $FLDR
    sleep .1 
    sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh vdaily_${i} compute_daily_v_1d.py $CFG_NAME false false $FLDR
#
    sleep .5
    sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh umonthly_${i} compute_monthly_u_1d.py $CFG_NAME false false $FLDR
    sleep .1
    sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh vmonthly_${i} compute_monthly_v_1d.py $CFG_NAME false false $FLDR
#    sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh KE8_${i} compute_KE_filtered_8.py $CFG_NAME false false $FLDR
#    sleep .5
#    sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh KE20_${i} compute_KE_filtered_20.py $CFG_NAME false false $FLDR
#    sleep .5
#    sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh KEall_${i} compute_KE_filtered_all_from_hourly_means.py $CFG_NAME false false $FLDR
#    sleep .5
 
    #sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh u20_${i} filter_u20.py $CFG_NAME false false $FLDR
    #sleep .5
    #sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh v20_${i} filter_v20.py $CFG_NAME false false $FLDR
    #sleep .5

    #sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh u36_${i} filter_u36.py $CFG_NAME false false $FLDR
    #sleep .5
    #sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh v36_${i} filter_v36.py $CFG_NAME false false $FLDR 
    #sleep .5
    #sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh u8_${i} filter_u8.py $CFG_NAME false false $FLDR
    #sleep .5
    #sbatch ${SCRIPT_DIR}/sed_and_launch_filt.sh v8_${i} filter_v8.py $CFG_NAME false false $FLDR
   i=$(( $i + 1 ))
   done
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
  ind=0

  cd $OUT_DIR
  for FLDR in $FLDR_VERSION1 $FLDR_VERSION2
  do
  if ${IS_AGRIF_LIST[${ind}]} ;
  then
  export VER=${VER_LIST[${ind}]}
  echo $VER 
     for script in $scripts_zoom ;
     do
        ${SCRIPT_DIR}/sed_and_launch_plots.sh $script $AGRIF_CFG_NAME true false $FLDR

     done
  ind=$(( $ind + 1 ))
  fi
  done

  if $DO_CMP_ZOOM ;
  then
      for script in $scripts_zoom_cmp ;
      do
         ${SCRIPT_DIR}/sed_and_launch_plots.sh $script $AGRIF_CFG_NAME true true $FLDR_VERSION1 $FLDR_VERSION2

      done
  fi
fi

################## make_gifs #######################
ind=0
VER_LIST=($VER1 $VER2)
if $DO_GIFS_MOTHER;
then
    cd $OUT_DIR
    mkdir GIFS
    cd GIFS/
    for FLDR in $FLDR_VERSION1 $FLDR_VERSION2
    do
    export VER=${VER_LIST[${ind}]}
    echo $VER 
       for script in $scripts_gifs ;
       do
          ${SCRIPT_DIR}/do_gifs.sh $script $CFG_NAME false false $FLDR

       done
       for script in $scripts_gifs_daily ;
       do
          ${SCRIPT_DIR}/do_gifs_daily.sh $script $CFG_NAME false false $FLDR

       done

    ind=$(( $ind + 1 ))
    done
fi
ind=0
if $DO_GIFS_ZOOM;
then
    cd $OUT_DIR
    mkdir GIFS
    cd GIFS/

    for FLDR in $FLDR_VERSION1 $FLDR_VERSION2
    do
    export VER=${VER_LIST[${ind}]}
    echo $VER 
       for script in $scripts_gifs ;
       do
          sbatch ${SCRIPT_DIR}/do_gifs.sh $script $AGRIF_CFG_NAME true false $FLDR 

       done
       for script in $scripts_gifs_daily ;
       do
          sbatch ${SCRIPT_DIR}/do_gifs_daily.sh $script $AGRIF_CFG_NAME true false $FLDR

       done

    ind=$(( $ind + 1 ))
    done
fi

if $COMBINE_NOOBS
then
   jobcombine=$(sbatch ${SCRIPT_DIR}/combine_ola.sub.zoom $CYCLE_NOOBS $FLDR_VERSION1 $FLDR_VERSION2)
   jobcombine_ID=$(echo $jobcombine  | awk '{print $4}')
fi

if $DO_PLOT_NOOBS;
then
    export VER=${VER_LIST[${ind}]}
    if $IS_AGRIF_VER1 
    then

       if $COMBINE_NOOBS
       then
       #sbatch ${SCRIPT_DIR}/plots_noobs.sub $CYCLE_NOOBS $FLDR_VERSION1 $FLDR_VERSION2
       jobplotnoobs1=$(sbatch --dependency=afterany:${jobcombine_ID} ${SCRIPT_DIR}/plots_noobs.sub.zoom $CYCLE_NOOBS $FLDR_VERSION1 $FLDR_VERSION2)
       jobplotnoobs1_ID=$(echo $jobplotnoobs1 | awk '{print $4}')

       else
       #jobplotnoobs1=$(sbatch ${SCRIPT_DIR}/plots_noobs.sub.zoom $CYCLE_NOOBS $FLDR_VERSION1 $FLDR_VERSION2)
       #jobplotnoobs1_ID=$(echo $jobplotnoobs1 | awk '{print $4}')
       #sbatch --dependency=afterany:${jobplotnoobs1_ID} ${SCRIPT_DIR}/plots_noobs.sub $CYCLE_NOOBS $FLDR_VERSION1 $FLDR_VERSION2
       ${SCRIPT_DIR}/plots_noobs.sub.zoom $CYCLE_NOOBS $FLDR_VERSION1 $FLDR_VERSION2
       #${SCRIPT_DIR}/plots_noobs.sub $CYCLE_NOOBS $FLDR_VERSION1 $FLDR_VERSION2
       fi     
       
    fi 
fi
