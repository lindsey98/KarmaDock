import os

import pandas as pd
import torch

if __name__ == '__main__':
    protein_dict = torch.load('/home/ruofan/git_space/TankBind/datasets/protein_315.pt') # this is processed by TankBind
    out_dir = '/home/ruofan/git_space/KarmaDock/datasets/TankBind_KaramDock_both'
    os.makedirs(out_dir, exist_ok=True)

    for proteinName in list(protein_dict.keys()):
        karmadock_results_csv = f"/home/ruofan/git_space/KarmaDock/datasets/protein315_to_drugbank9k_results_csv/protein315_to_drugbank9k_{proteinName}_results.csv"
        karmadock_df = pd.read_csv(karmadock_results_csv)
        karmadock_df['compound_name'] = [x+'_rdkit' for x in list(karmadock_df['compound_name'])]

        tankbind_results_csv = f"/home/ruofan/git_space/TankBind/datasets/protein315_to_drugbank9k_results_csv/protein315_to_drugbank9k_{proteinName}_results.csv"
        tankbind_df = pd.read_csv(tankbind_results_csv, index_col=0)

        merged_df = pd.merge(karmadock_df, tankbind_df)
        chosen_df = merged_df.query("affinity > 7 & binding_strength > 50").reset_index(drop=True)
        if len(chosen_df) > 0:
            chosen_df.to_csv(os.path.join(out_dir, f"{proteinName}.csv"), index=False)
