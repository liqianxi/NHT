from nht.utils import project_d4rl_actions

d4rl_dset = "walker2d-expert-v2"
NHT_path = '.results/walker/20_multihead/version_0'
dset_with_projected_actions = project_d4rl_actions(NHT_path, d4rl_dset, prop=0.01)

for i in range(5):
    a = dset_with_projected_actions['actions'][i]
    print(a)