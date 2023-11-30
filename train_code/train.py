# -*- coding: utf-8 -*-

import argparse
import os, shutil, sys
import time
import warnings

warnings.filterwarnings("ignore")

# import from local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt


def storage_manage():
    if not os.path.exists("runs_last/"):
        os.mkdir("runs_last/")
    
    # copy to the new address
    new_address = "runs_last/"+str(int(time.time()))+"/"
    shutil.copytree("runs/", new_address)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto_resume_closest', action='store_true')
    parser.add_argument('--auto_resume_best', action='store_true')
    parser.add_argument('--use_pretrained', action='store_true')
    # parser.add_argument('--resume_net', action='store_true', help="GAN use pre-trained esr-net as initial weight")
    parser.add_argument('--clean_best', action='store_true')
    parser.add_argument('--platform', type=str, default="PC")

    global args
    args = parser.parse_args()


    if args.auto_resume_closest and args.auto_resume_best:
        print("you could only resume either nearest or best, not both")
        os._exit(0)
    
    if not args.auto_resume_closest and not args.auto_resume_best:
        if os.path.exists("./runs"):
            storage_manage()
            shutil.rmtree("./runs")



def folder_prepare():
    def _make_folder(folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def _delete_and_make_folder(folder_name):
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    
    # The lists we care about
    make_folder_name_lists = ["saved_models/", "saved_models/checkpoints/", "datasets/"]
    delete_and_make_folder_name_lists = []

    for folder_name in make_folder_name_lists:
        _make_folder(folder_name)

    for folder_name in delete_and_make_folder_name_lists:
        _delete_and_make_folder(folder_name)

    

def process(options):
    print(args)
    start = time.time()

    # Switch based on the model architecture
    if options['architecture'] == "GRL":
        from train_grl import train_grl
        obj = train_grl(options, args)
    elif options['architecture'] == "GRLGAN":
        from train_grlgan import train_grlgan
        obj = train_grlgan(options, args)
    else:
        raise NotImplementedError("This is not a supported model architecture")


    obj.run()

    total_time = time.time() - start
    print("All programs spent {} hour {} min {} s".format(str(total_time//3600), str((total_time%3600)//60), str(total_time%3600)))


def main():
    parse_args()

    folder_prepare()
    process(opt)

if __name__ == "__main__":
    main()