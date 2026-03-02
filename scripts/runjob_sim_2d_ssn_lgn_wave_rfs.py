import os
import argparse
import numpy as np
from subprocess import Popen
import time
from sys import platform
import uuid
import random

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
    parser.add_argument('--n_e', '-ne', help='number of excitatory cells',type=int, default=1)
    parser.add_argument('--n_i', '-ni', help='number of inhibitory cells',type=int, default=1)
    parser.add_argument('--init_iter', '-iit', help='initial iteration number',type=int, default=0)
    parser.add_argument('--batch_iter', '-bit', help='number of iterations to run per batch',type=int, default=100)
    parser.add_argument('--max_iter', '-mit', help='max iteration number',type=int, default=100)
    parser.add_argument('--s_x', '-sx', help='feedforward arbor decay length',type=float, default=0.08)
    parser.add_argument('--s_s', '-ss', help='retinotopic scatter decay length',type=float, default=0.00)
    parser.add_argument('--gain_i', '-gi', help='gain of inhibitory cells',type=float, default=1.0)
    parser.add_argument('--wff_sum', '-w', help='feedforward weight strength',type=float, default=1.0)
    # parser.add_argument('--hebb_wei', '-hei', help='whether wei has Hebbian learning rule',type=int, default=0)
    # parser.add_argument('--hebb_wii', '-hii', help='whether wii has Hebbian learning rule',type=int, default=0)
    parser.add_argument('--prune', '-p', help='whether to prune feedforward weights',type=int, default=0)
    parser.add_argument('--rec_plast', '-r', help='whether recurrent weights are plastic',type=int, default=0)
    parser.add_argument('--rec_i_ltd', '-d', help='factor for inhibitory LTD term',type=float, default=1.0)
    parser.add_argument('--seed', '-s', help='seed',type=int, default=0)
    parser.add_argument('--n_wave', '-nw', help='number of geniculate waves',type=int, default=15)
    parser.add_argument('--n_stim', '-ns', help='number of light/dark sweeping bars',type=int, default=2)
    parser.add_argument('--n_shrink', '-nh', help='factor by which to shrink stimuli',type=float, default=1.0)
    parser.add_argument('--n_grid', '-ng', help='number of points per grid edge',type=int, default=20)
    parser.add_argument('--mode', '-m', help='mode',type=str, default='spont_vis')
    parser.add_argument('--gb', '-g', help='number of gbs per cpu',type=int, default=6)
    
    args2 = parser.parse_args()
    args = vars(args2)
    
    cluster = str(args["cluster_"])
    n_e = int(args['n_e'])
    n_i = int(args['n_i'])
    init_iter = int(args['init_iter'])
    batch_iter = int(args['batch_iter'])
    max_iter = int(args['max_iter'])
    s_x = args['s_x']
    s_s = args['s_s']
    gain_i = args['gain_i']
    wff_sum = args['wff_sum']
    # hebb_wei = int(args['hebb_wei'])
    # hebb_wii = int(args['hebb_wii'])
    prune = int(args['prune'])
    rec_plast = int(args['rec_plast'])
    rec_i_ltd = args['rec_i_ltd']
    seed = int(args['seed'])
    n_wave = int(args['n_wave'])
    n_stim = int(args['n_stim'])
    n_shrink = args['n_shrink']
    n_grid = int(args['n_grid'])
    mode = str(args['mode'])
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
    
    #--------------------------------------------------------------------------
    # Make SBTACH
    inpath = currwd + "/sim_2d_ssn_lgn_wave_rfs.py"
    c1 = "{:s} -ne {:d} -ni {:d} -iit {:d} -bit {:d} -mit {:d} -s {:d} -nw {:d} -ns {:d} -nh {:.2f} -ng {:d} -sx {:.2f} -ss {:.2f} -gi {:.1f} -w {:.2f} -p {:d} -r {:d} -d {:f} -m {:s}".format(
        inpath,n_e,n_i,init_iter,batch_iter,max_iter,seed,n_wave,n_stim,n_shrink,n_grid,s_x,s_s,gain_i,wff_sum,prune,rec_plast,rec_i_ltd,mode)
    
    jobname="{:s}".format('sim_2d_lgn_{:s}_rfs_s_{:d}_n_{:d}_sx={:.2f}_ss={:.2f}_gi={:.1f}_wff={:.1f}_p={:d}_r={:d}_d={:.1f}'.format(
        mode,seed,init_iter,s_x,s_s,gain_i,wff_sum,prune,rec_plast,rec_i_ltd))
    
    if not args2.test:
        jobnameDir=os.path.join(ofilesdir, jobname)
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


