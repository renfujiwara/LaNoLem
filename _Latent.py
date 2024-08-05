import argparse
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["VECLIB_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
os.environ["NUMBA_NUM_THREADS"]="1"

from model import NLDS, make_data, plot_result


def main(args):
    dataset_name = args.dataset_name
    data, setting = make_data(dataset_name, latent=True, noise_ratio=0.0)
    model = NLDS(verbose=False, num_works=args.num_works, print_log=True)
    model.random_state = 42
    model = model.fit(data, fit_type='Latent', fit_init=True)
    plot_result(model, data, setting, fsize=3.3)
    
            
def parse_args():
    desc = "hyperparameter tuning"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset_name', default=None, required=True, type=str, help='string to identify experiment')
    parser.add_argument('--num_works', default=-1, type=int, help='num process')
    parser.add_argument('--obs_dim', default=5, type=int, help='num process')
    return parser.parse_args()

if __name__ == "__main__":
    print(os.getpid())
    args = parse_args()
    if args is None:
        exit()
    print('main_start', flush=True)
    main(args)
    print('main_end', flush=True)