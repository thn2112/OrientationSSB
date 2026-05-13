import os
import argparse
import numpy as np
from subprocess import Popen
import time
from sys import platform
import uuid
import random
from tempfile import TemporaryDirectory

from importlib import reload


def runjobs():


    """
        Function to be run in a Sun Grid Engine queuing system. For testing the output, run it like
        python runjobs.py --test 1
        
    """
    
    #--------------------------------------------------------------------------
    # Test commands option
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=int, default=0)
    parser.add_argument("--cluster_", help=" String", default='burg')
    parser.add_argument('--n_ori', '-no', help='number of orientations',type=int, default=8)
    parser.add_argument('--n_phs', '-np', help='number of orientations',type=int, default=8)
    # parser.add_argument('--n_rpt', '-nr', help='number of repetitions per orientation',type=int, default=5)
    parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=2)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])
    n_ori = int(args['n_ori'])
    n_phs = int(args['n_phs'])
    # n_rpt = int(args['n_rpt'])
    gb = int(args['gb'])

    
    if (args2.test):
        print ("testing commands")
    
    #--------------------------------------------------------------------------
    # Which cluster to use

    
    if platform=='darwin':
        cluster='local'
    
    currwd = os.getcwd()

    #--------------------------------------------------------------------------
    # Ofiles folder

    user = os.environ["USER"]
        
    if cluster=='haba':
        path_2_package="/rigel/theory/users/"+user+"/OrientationSSB/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"

    if cluster=='moto':
        path_2_package="/moto/theory/users/"+user+"/OrientationSSB/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"

    if cluster=='burg':
        path_2_package="/burg/theory/users/"+user+"/OrientationSSB/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"
        
    elif cluster=='axon':
        path_2_package="/home/"+user+"/OrientationSSB/scripts"
        ofilesdir = path_2_package + "/Ofiles/"
        resultsdir = path_2_package + "/results/"
        
    elif cluster=='local':
        path_2_package="/Users/tuannguyen/OrientationSSB/scripts"
        ofilesdir = path_2_package+"/Ofiles/"
        resultsdir = path_2_package + "/results/"


    if not os.path.exists(ofilesdir):
        os.makedirs(ofilesdir)

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    time.sleep(0.2)
    
    maps = ['','band']#,'band_4','band_8','band_12','band_16']
    seeds = range(20)

    with TemporaryDirectory() as temp_dir:
        for map_type in maps:
            if map_type == '':
                static_phase_orisel_sandp_ffl4s = [(0,0,0,0,0),(1,0,0,0,0),
                                                   (0,1,0,0,0),(1,1,0,0,0)
                                                   (0,0,1,0,0),(0,0,0,1,0),
                                                   (0,0,0,0,1),(1,0,0,0,1)]
            else:
                static_phase_orisel_sandp_ffl4s = [(0,0,0,0,0)]
            for (static,phase,orisel,sandp,ffl4) in static_phase_orisel_sandp_ffl4s:
                for seed in seeds:
                    #--------------------------------------------------------------------------
                    # Make SBTACH
                    inpath = currwd + "/sim_L4_act_L23_sel.py"
                    c1 = "{:s} -s {:d} -no {:d} -np {:d} -r 1".format(
                            inpath,seed,n_ori,n_phs)
                    res_dir = './../results/L23_sel/'
                    if static == 1:
                        c1 = c1 + " -st 1"
                        res_dir = res_dir + 'static_'
                    if map_type != '':
                        c1 = c1 + " -m {:s}".format(map_type)
                        res_dir = res_dir + '{:s}_'.format(map_type)
                    if phase == 1:
                        c1 = c1 + " -ap 1"
                        res_dir = res_dir + 'phase_'
                    if orisel == 1:
                        c1 = c1 + " -aos 1"
                        res_dir = res_dir + 'orisel_'
                    if sandp == 1:
                        c1 = c1 + " -asp 1"
                        res_dir = res_dir + 'sandp_'
                    if ffl4 == 1:
                        c1 = c1 + " -aff 1"
                        res_dir = res_dir + 'ffl4_'
                    if os.path.isfile(res_dir+'seed={:d}.pkl'.format(seed)):
                        continue

                    jobname="{:s}_map={:s}_static={:d}_phase={:d}_orisel={:d}_sandp={:d}_ffl4={:d}_seed={:d}".format(
                        'sim_L4_act_L23_sel',map_type,static,phase,orisel,sandp,ffl4,seed)

                    if not args2.test:
                        jobnameDir=os.path.join(temp_dir, jobname)
                        text_file=open(jobnameDir, "w");
                        os. system("chmod u+x "+ jobnameDir)
                        text_file.write("#!/bin/sh \n")
                        if cluster=='haba' or cluster=='moto' or cluster=='burg':
                            text_file.write("#SBATCH --account=theory \n")
                        text_file.write("#SBATCH --job-name="+jobname+ "\n")
                        text_file.write("#SBATCH -t 0-11:59  \n")
                        text_file.write("#SBATCH --mem-per-cpu={:d}gb \n".format(gb))
                        # text_file.write("#SBATCH --gres=gpu\n")
                        text_file.write("#SBATCH -c 1 \n")
                        text_file.write("#SBATCH -o "+ ofilesdir + "/%x.%j.o # STDOUT \n")
                        text_file.write("#SBATCH -e "+ ofilesdir +"/%x.%j.e # STDERR \n")
                        text_file.write("python  -W ignore " + c1+" \n")
                        text_file.write("echo $PATH  \n")
                        text_file.write("exit 0  \n")
                        text_file.close()

                        if cluster=='axon':
                            os.system("sbatch -p burst " +jobnameDir);
                        else:
                            os.system("sbatch " +jobnameDir);
                    else:
                        print (c1)


if __name__ == "__main__":
    runjobs()


