#!/bin/sh
user=$1

if [ $user == "tbrivoalperso" ];
then
    echo "signin in as tbrivoalperso" 
    git config user.name "tbrivoalperso"
    git config user.email "theo.brivoal@gmail.com"
else
    git config user.name "tbrivoal"
    git config user.email "theo.brivoal@mercator-ocean.fr"
fi

