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
    parser.add_argument('--n_patt', '-npt', help='number of spontaneous patterns',type=int, default=50)
    # parser.add_argument('--n_rpt', '-nr', help='number of repetitions per orientation',type=int, default=5)
    parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=2)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])
    n_patt = int(args['n_patt'])
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
    
    num_seeds = 20
    pcvs = [0.5,0.65,0.8,1.0]

    with TemporaryDirectory() as temp_dir:
        for pcv in pcvs:
            #--------------------------------------------------------------------------
            # Make SBTACH
            inpath = currwd + "/analyze_L23_spont.py"
            c1 = "{:s} -s {:d} -npt {:d} -pcv {:.2f}".format(
                    inpath,num_seeds,n_patt,pcv)

            jobname="{:s}_cv={:.2f}".format(
                'analyze_L23_spont',pcv)
            
            if not args2.test:
                jobnameDir=os.path.join(temp_dir, jobname)
                text_file=open(jobnameDir, "w");
                os. system("chmod u+x "+ jobnameDir)
                text_file.write("#!/bin/sh \n")
                if cluster=='haba' or cluster=='moto' or cluster=='burg':
                    text_file.write("#SBATCH --account=theory \n")
                text_file.write("#SBATCH --job-name="+jobname+ "\n")
                text_file.write("#SBATCH -t 0-2:59  \n")
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


