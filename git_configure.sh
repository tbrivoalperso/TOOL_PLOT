#!/bin/sh
user=$1

if [ $user == "tbrivoalperso" ];
then
    echo "signin in as tbrivoalperso" 
    git remote remove origin
    git config user.name "tbrivoalperso"
    git config user.email "theo.brivoal@gmail.com"
    git remote add origin git@github.com-tbrivoalperso:tbrivoalperso/TOOL_PLOT.git
else
    ssh -T git@github.com-tbrivoal
    git config user.name "tbrivoal"
    git config user.email "theo.brivoal@mercator-ocean.fr"
fi

