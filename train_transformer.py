import os
import yaml
from trainer import main_worker

import argparse


parser = argparse.ArgumentParser(description='CO2 Estimation (Time Series Model)')

parser.add_argument("--cfg", metavar="Config Filename", default="train_co2_estimation_W18", 
                    help="Experiment to run. Default is train_co2_estimation_W18")
parser.add_argument("--ws", type=int, default=10, help="window_size.")
                    
def run_task(config, args):

    config['model']['window_size'] = args.ws
    model_path = os.path.join(config['checkpoint']['ckpt_path']).format(config['model']['type'], config['model']['window_size'])
    print("ckpt path:", model_path)

    # Simply call main_worker function
    main_worker(config, args)

    # # Simply call main_worker function
    # main_worker(config, args)
        
def main():
    args = parser.parse_args()
    cfg = args.cfg if args.cfg[-5:] == '.yaml' else args.cfg + '.yaml'
    config_path = os.path.join(os.getcwd(), 'cfg', cfg)
    assert os.path.exists(config_path), f"Could not find {cfg} in configs directory!"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    
    print("=> Config Details")
    print(config) #For reference in logs
    
    run_task(config, args)

if __name__ == "__main__":
    main()