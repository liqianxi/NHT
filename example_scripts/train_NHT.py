import sys
from nht.utils import action_map_arg_parser
from nht.train_action_map import train_action_map

arg_list = [
    "--max_epochs", "3", 
    "--d4rl_dset", "walker2d-expert-v2",
    "--default_root_dir", ".results/walker",
    "--lr", "0.0001",
    "--rnd_seed", "101",
    "--run_name", "20_multihead",
    "--hiddens", "128,128,128",
    #"--accelerator", "gpu",
    "--accelerator", "cpu",
    #"--devices", "1",
    "--a_dim", "2", 
    "--context", "observations", 
    "--model", "NHT",
    "--multihead",
    "--lipschitz_coeff", "20"
]

sys.argv.extend(arg_list)

parser = action_map_arg_parser()
args = parser.parse_args()
print(args)

# train action map
model = train_action_map(args)