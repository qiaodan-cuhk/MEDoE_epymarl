import json
import pandas as pd




"""
# chainball
alg = "qmix_ns"
env = "chainball:chainball-v0"
save_name="qmix_chainball.pkl"
run_ids = list(range(1,227)) #
filter_cfg = {}
#filter_cfg = {"lr": 0.0003}
scale_fn = lambda x: (x+84.3)/(16.3+84.3)
is_sweep = True
sweep_save_name="qmix_chainball_sweep.pkl"
"""

"""
# overcooked
alg = "qmix_ns"
env = "gym_cooking:cookingSplit-v0"
save_name="qmix_cooking.pkl"
run_ids = list(range(1,33))
filter_cfg = {}
scale_fn = lambda x: x/2
is_sweep = True
sweep_save_name="qmix_cooking_sweep.pkl"
"""

# vmas
alg = "qmix_ns"
env = "vmas.gym:VMASFootball-2-v0"
save_name="qmix_vmas.pkl"
run_ids = list(range(1,33)) #
filter_cfg = {}
scale_fn = lambda x: x/4
is_sweep = True
sweep_save_name="qmix_vmas_sweep.pkl"


runs = {}
configs = {}
for run_id in run_ids:

    # Check config
    config_fname = f"sacred/{alg}/{env}/{run_id}/config.json"
    with open(config_fname, 'r') as j:
        config = json.loads(j.read())
    if not all(config[key]==val for key,val in filter_cfg.items()):
        # Skip if doesn't match the config filter
        continue

    metrics_fname = f"sacred/{alg}/{env}/{run_id}/metrics.json"
    with open(metrics_fname, 'r') as j:
        try:
            metrics = json.loads(j.read())
        except:
            print(f"Issue with {config_fname}")
    config_fname = f"sacred/{alg}/{env}/{run_id}/config.json"
    with open(config_fname, 'r') as j:
        try:
            config = json.loads(j.read())
        except:
            print(f"Issue with {config_fname}")

    runs[run_id] = {
        #"lr": config["lr"],
        #"target_update_interval_or_tau": config["target_update_interval_or_tau"],
        #"epsilon_anneal_time": config["epsilon_anneal_time"],
        #"gamma": config["gamma"],
        "steps": metrics["return_mean"]["steps"],
        "values": metrics["return_mean"]["values"],
        }
    configs[run_id] = {
        "lr": config["lr"],
        "target_update_interval_or_tau": config["target_update_interval_or_tau"],
        "epsilon_anneal_time": config["epsilon_anneal_time"],
        "gamma": config["gamma"],
        }

dfs = []
for key, value in runs.items():
    df = pd.DataFrame(value)
    df = df.set_index("steps")
    df = df.rename(columns={"values": key})
    if len(dfs) > 0:
        df = df.reindex(dfs[0].index, method="nearest")
    dfs.append(df)
final_df = pd.concat(dfs, axis=1)
final_df_melted = final_df.reset_index().melt(id_vars="steps", var_name="run", value_name="values")
final_df_melted.rename(columns={"steps": "Train/step", "values": "Return/team/mean"}, inplace=True)
final_df_melted.drop(columns=["run"], inplace=True)
final_df_melted["Method"] = "nn-QMIX"
final_df_melted["ID"] = final_df_melted.index
final_df_melted.set_index(["Method", "ID"], inplace=True)
output_df = final_df_melted.apply(lambda y: scale_fn(y) if y.name=="Return/team/mean" else y)
output_df.to_pickle(save_name)

# here let's try to make a dataframe which has the sweep params and final return (and perhaps later AUC)
if is_sweep:
    final_returns = final_df.iloc[-1].apply(lambda y: scale_fn(y)).rename("final return")
    configs = pd.DataFrame(configs).transpose()
    sweep_df = pd.concat((configs, final_returns), axis=1).sort_values("final return", ascending=False)
    sweep_df.to_pickle(sweep_save_name)
    print(sweep_df)


print(final_df.mean(axis=1).apply(scale_fn)) # print the results
print(output_df)
