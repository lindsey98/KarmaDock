
# here put the import lib
import argparse
import os
import sys

import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim
from prefetch_generator import BackgroundGenerator
import warnings
from tqdm import tqdm

from utils.fns import Early_stopper, set_random_seed
from dataset.graph_obj import VSTestGraphDataset_Fly_SMI, get_mol2_xyz_from_cmd
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.KarmaDock_architecture import KarmaDock
from utils.post_processing import correct_one_both_postprocessing, correct_one
from utils.pre_processing import get_pocket_pure, read_mol
import rdkit.Chem as Chem
from Bio.PDB import PDBParser
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
os.environ['CUBLAS_WORKSPACE_CONFIG']=":16:8"

class DataLoaderX(PassNoneDataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_calpha_center_coords(pdb_filename):

    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb_filename, pdb_filename)
    res_list = list(s.get_residues())
    res_list = get_clean_res_list(res_list, ensure_ca_exist=True)

    res_list = [res for res in res_list if \
                    (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))] # only those residues that contain the four essential backbone atoms: Nitrogen (N), Carbon Alpha (CA), Carbon (C), and Oxygen (O)
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]: # N、CA、C 和 O 这四种原子构成了蛋白质主链的骨架
            res_coords.append(list(atom.coord))
        coords.append(res_coords)

    coords = torch.as_tensor(coords, dtype=torch.float32)
    mask = torch.isfinite(coords.sum(dim=(1, 2)))
    coords[~mask] = np.inf

    X_ca = coords[:, 1]  # 只看α碳
    mean_coords = X_ca.mean(axis=0).numpy()
    return mean_coords[np.newaxis, :]

@torch.inference_mode()
def predict_single_protein(proteinName,
                           model,
                           device,
                           result_csv_prefix="protein5_to_drugbank9k",
                           num_trails=5):
    # Protein PDB
    protein_pdbfile = f"{args.proteinDirsUni}/{proteinName}/{proteinName}.pdb"

    ### P2rank predicted pockets
    p2rankFile = f"{args.p2rank_pred_dir}/{proteinName}.pdb_predictions.csv"
    pocket = pd.read_csv(p2rankFile)
    pocket.columns = pocket.columns.str.strip()
    pocket_coords = pocket[['center_x', 'center_y', 'center_z']].values  # get all predicted pocket centers

    search_pattern = os.path.join(args.protein_pockets_dir, f"{proteinName}_pocket_*.pdb")
    pocket_files = glob.glob(search_pattern)
    print(f"Found {len(pocket_files)} pockets for {proteinName}.")

    for trail in range(1, num_trails+1):

        protein_name = []
        compound_name = []
        binding_scores = []
        pocket_names = []
        pocket_coms = []

        if os.path.exists(f"{args.out_dir}/{result_csv_prefix}_{proteinName}_results_trail{trail}.csv"):
            continue

        for i, pocket_file in enumerate(pocket_files):
            pattern = r"/([^/]+)_(pocket_(?:center|\d+))\.pdb$"
            match = re.search(pattern, pocket_file)
            pocketName = match.group(2)  #

            if pocketName.split('pocket_')[1] == 'center':
                pocket_center = get_calpha_center_coords(protein_pdbfile) # pocket_center is protein center
                pocket_center = torch.from_numpy(pocket_center)
                try:
                    test_dataset = VSTestGraphDataset_Fly_SMI(protein_file=pocket_file,
                                                              ligand_path=args.ligand_smi,
                                                              pocket_center=pocket_center,
                                                              protein_name=proteinName)
                except EOFError as e:
                    print(e, proteinName)
                    continue
            else:
                pocket_number = int(pocketName.split('pocket_')[1]) - 1
                pocket_center = torch.Tensor(pocket_coords[pocket_number])
                try:
                    test_dataset = VSTestGraphDataset_Fly_SMI(protein_file=pocket_file,
                                                              ligand_path=args.ligand_smi,
                                                              pocket_center=pocket_center,
                                                              protein_name=proteinName)
                except EOFError as e:
                    print(e, proteinName)
                    continue

            test_dataset.generate_graphs(ligand_smi=args.ligand_smi, n_job=-1) # use CPU to generate graph -> this may take time
            protein_name_pocket, compound_name_pocket, binding_scores_pocket, pocket_names_pocket = make_inference(test_dataset=test_dataset,
                                                                                                                  model=model,
                                                                                                                  device=device,
                                                                                                                  proteinname=proteinName,
                                                                                                                  pocketname=pocketName,
                                                                                                                   out_dir=os.path.join(args.out_vis_dir, proteinName),
                                                                                                                   postprocessing=False
                                                                                                                   )
            protein_name.extend(protein_name_pocket)
            compound_name.extend(compound_name_pocket)
            binding_scores.extend(binding_scores_pocket)
            pocket_names.extend(pocket_names_pocket)
            pocket_coms.extend([",".join([str(a.round(3)) for a in pocket_center.numpy()])]*len(protein_name_pocket))

        df_score = pd.DataFrame(list(zip(protein_name, compound_name, pocket_names, pocket_coms, binding_scores)),
                                columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com', 'binding_strength'])

        df_score.to_csv(f"{args.out_dir}/{result_csv_prefix}_{proteinName}_results_trail{trail}.csv", index=False)

@torch.inference_mode()
def make_inference(test_dataset, model, device, proteinname, pocketname, out_dir, postprocessing=False):
    protein_name = []
    compound_name = []
    binding_scores = []
    pocket_names = []

    os.makedirs(out_dir, exist_ok=True)
    # dataloader
    test_dataloader = DataLoaderX(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  follow_batch=[],
                                  pin_memory=True)

    for idx, data in enumerate(tqdm(test_dataloader)):
        data = data.to(device)
        batch_size = data['ligand'].batch[-1] + 1
        # forward
        pro_node_s, lig_node_s = model.module.encoding(data)
        lig_pos, _, _ = model.module.docking(pro_node_s, lig_node_s, data, recycle_num=3)
        mdn_score_pred = model.module.scoring(lig_s=lig_node_s,
                                              lig_pos=lig_pos,
                                              pro_s=pro_node_s,
                                              data=data,
                                              dist_threhold=5.,
                                              batch_size=batch_size)
        if postprocessing: # this will further correct the pose using force-field (FF) optimization or aligning the predicted conformation with the rational conformation generated by RDkit.
            data.pdb_id = [x + f'_{pocketname}' for x in data.pdb_id]
            poses, _ = post_processing(
                          lig_pos=lig_pos,
                          data=data,
                          out_dir=out_dir,
                            method='ff')
            corrected_pos = torch.from_numpy(np.concatenate((poses))).to(device)
            mdn_score_pred_corrected = model.module.scoring(lig_s=lig_node_s,
                                                               lig_pos=corrected_pos,
                                                               pro_s=pro_node_s,
                                                               data=data,
                                                               dist_threhold=5.,
                                                               batch_size=batch_size)
            protein_name.extend([proteinname] * batch_size)
            compound_name.extend(data.pdb_id)
            pocket_names.extend([pocketname] * batch_size)
            binding_scores.extend([round(score, 4) for score in mdn_score_pred_corrected.cpu().numpy().tolist()])
        else:
            protein_name.extend([proteinname] * batch_size)
            compound_name.extend(data.pdb_id)
            pocket_names.extend([pocketname] * batch_size)
            binding_scores.extend([round(score, 4) for score in mdn_score_pred.cpu().numpy().tolist()])

    return protein_name, compound_name, binding_scores, pocket_names

@torch.inference_mode()
def post_processing(lig_pos, data, out_dir, method, out_corrected=True, addHs=True):
    # # post processing
    data.pos_preds = lig_pos
    poses = []

    for idx, mol in enumerate(data['ligand'].mol):
        # correct pos
        pos_pred = data.pos_preds[data['ligand'].batch == idx].cpu().numpy().astype(np.float64)  # + pocket_centers[idx]
        start_time = time.perf_counter()
        if method == 'ff':
            corrected_mol, uncorrected_mol = correct_one(mol=mol, pos_pred=pos_pred, method='ff')
        elif method == 'align':
            corrected_mol, uncorrected_mol = correct_one(mol=mol, pos_pred=pos_pred, method='align')
        else:
            raise NotImplementedError
        # else:
        #     corrected_mol, uncorrected_mol = correct_one_both_postprocessing(mol=mol, pos_pred=pos_pred)

        postprocessing_time = time.perf_counter()
        poses.append(corrected_mol.GetConformer().GetPositions())

        if out_corrected:
            corrected_file = f'{out_dir}/{data.pdb_id[idx]}_pred_{method}_corrected.sdf'
            try:
                if addHs:
                    corrected_mol = Chem.AddHs(corrected_mol, addCoords=True)
                Chem.MolToMolFile(corrected_mol, corrected_file)
            except:
                print(f'save {corrected_file} failed')
                pass

    return poses, postprocessing_time - start_time

if __name__ == '__main__':

    # get parameters from command line
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--out_dir', type=str,
                           default='./datasets/protein5_to_drugbank9k_results_csv',
                           help='dir for recording binding poses and binding scores')
    # argparser.add_argument('--out_vis_dir', type=str,
                           # default='./datasets/protein5_to_drugbank9k_vis',
                           # help='dir for visualization')
    argparser.add_argument('--out_vis_dir', type=str,
                           default='./datasets/protein5_to_drugbank9k_vis_source',
                           help='dir for visualization')

    argparser.add_argument('--score_threshold', type=float,
                           default=72,
                           help='score threshold for saving binding poses')
    argparser.add_argument('--batch_size', type=int,
                           default=64,
                           help='batch size')
    argparser.add_argument('--random_seed', type=int,
                           default=2020,
                           help='random_seed')
    argparser.add_argument('--out_init', action='store_true', default=False, help='whether to save initial poses to sdf file')

    argparser.add_argument('--out_uncoorected', action='store_true', default=False, help='whether to save uncorrected poses to sdf file')
    argparser.add_argument('--out_corrected', action='store_true', default=False, help='whether to save corrected poses to sdf file')

    # argparser.add_argument('--proteinDirsUni', default="./datasets/FiveIsoformAlphafold", help='Directory for protein files (unfied)')
    argparser.add_argument('--proteinDirsUni', default="./datasets/FiveIsoformAlphafold_Source", help='Directory for protein files (unfied)')

    # argparser.add_argument('--p2rank_pred_dir', default="./datasets/protein_5_p2rank",
    #                        help='Dir to save p2rank predictions')

    argparser.add_argument('--p2rank_pred_dir', default="./datasets/protein_5_source_p2rank",
                           help='Dir to save p2rank predictions')

    # argparser.add_argument('--protein_pockets_dir', type=str,
    #                        default='./datasets/protein_5_segmented',
    #                        help='the protein files path')
    argparser.add_argument('--protein_pockets_dir', type=str,
                           default='./datasets/protein_5_source_segmented',
                           help='the protein files path')

    argparser.add_argument('--ligandDirs', default="/home/ruofan/git_space/TankBind/datasets/drugbank", help='Directory for ligand files')
    argparser.add_argument('--ligand_smi', type=str, default='./datasets/drugbank_9k.smi', help='Where to save the ligands smile')

    argparser.add_argument('--modelFile', default="./trained_models/karmadock_screening.pkl", help='Pretrained model file path')

    argparser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')


    args = argparser.parse_args()
    set_random_seed(args.random_seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_vis_dir, exist_ok=True)
    os.makedirs(args.p2rank_pred_dir, exist_ok=True)
    os.makedirs(args.protein_pockets_dir, exist_ok=True)


    '''p2rank predictions only run ONCE
    '''
    # with open('./protein_5_filelist.ds' , "w") as out:
    #     for proteinName in os.listdir(args.proteinDirsUni):
    #         # out.write(f"{args.proteinDirsUni}/{proteinName}/{proteinName}.pdb\n")
    #         out.write(f"{args.proteinDirsUni}/{proteinName}\n")
    #
    # cmd = f"bash ./p2rank_2.3/prank predict ./protein_5_filelist.ds -o {args.p2rank_pred_dir} -threads 1" ## fixme: you can increase thread
    # os.system(cmd)
    # exit()

    '''Aggregate all predicted pocket centers (as well as the protein center) as potential binding sites, and expand a radius of 20A as the potential binding pockets'''
    # for proteinName in os.listdir(args.proteinDirsUni):
    #     proteinName = proteinName.split('.pdb')[0]
    #     # protein_pdbfile = f"{args.proteinDirsUni}/{proteinName}/{proteinName}.pdb"
    #     protein_pdbfile = f"{args.proteinDirsUni}/{proteinName}.pdb"
    #
    #     ### read the p2rank prediction results, get all predicted pocket centers
    #     p2rankFile = f"{args.p2rank_pred_dir}/{proteinName}.pdb_predictions.csv"
    #     pocket = pd.read_csv(p2rankFile)
    #     pocket.columns = pocket.columns.str.strip()
    #     pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values # get all predicted pocket centers
    #
    #     ### get the protein center as a pocket as well
    #     protein_center = get_calpha_center_coords(protein_pdbfile)
    #
    #     ### select the corresponding pocket according to the pocket center, use radius=20 as TankBind does
    #     pocket_file = os.path.join(args.protein_pockets_dir, proteinName+'_pocket_center.pdb')
    #     if not os.path.exists(pocket_file):
    #          get_pocket_pure(protein_pdbfile,
    #                         somepoint=protein_center,
    #                         out_file=pocket_file,
    #                         size=20)
    #
    #     for i, com in enumerate(pocket_coms):
    #         pocket_file = os.path.join(args.protein_pockets_dir, proteinName + f'_pocket_{i+1}.pdb')
    #         if not os.path.exists(pocket_file):
    #             get_pocket_pure(protein_pdbfile,
    #                             somepoint=np.asarray([com]),
    #                             out_file=pocket_file,
    #                             size=20)
    # exit()

    '''Write all ligands in a single smile file'''
    # if (not os.path.exists(args.ligand_smi)) or len(open(args.ligand_smi).read()) == 0:
    #     compound_info = []
    #     for ligand in tqdm(os.listdir(args.ligandDirs)):
    #         ligandName = ligand.split('.sdf')[0]
    #         ligandFile = os.path.join(args.ligandDirs, ligand)
    #         mol, error = read_mol(ligandFile, None)  # unreadable by rdkit
    #         if error:
    #             continue
    #         if mol.GetNumAtoms() < 2: # single atom?
    #             continue
    #
    #         smiles = Chem.MolToSmiles(mol)
    #         compound_info.append((ligandName, smiles))
    #
    #     with open(args.ligand_smi, 'w') as f:
    #         for compound in compound_info:
    #             f.write(compound[0]+' '+compound[1]+'\n')
    # exit()

    '''Make predictions'''
    # load model
    device = args.device
    model = KarmaDock()
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(device)
    model.load_state_dict(torch.load(args.modelFile, map_location=device)['model_state_dict'],
                          strict=False)
    model.eval()
    # #
    # for proteinName in os.listdir(args.proteinDirsUni):
    #     predict_single_protein(proteinName, result_csv_prefix="protein5_to_drugbank9k")


    '''Pair validation with post-processing'''
    # Protein PDB
    proteinName = "UBA2"
    pocketName = "pocket_22"  #

    ### P2rank predicted pockets
    p2rankFile = f"{args.p2rank_pred_dir}/{proteinName}.pdb_predictions.csv"
    pocket = pd.read_csv(p2rankFile)
    pocket.columns = pocket.columns.str.strip()
    pocket_coords = pocket[['center_x', 'center_y', 'center_z']].values  # get all predicted pocket centers

    pocket_number = int(pocketName.split('pocket_')[1]) - 1
    pocket_center = torch.Tensor(pocket_coords[pocket_number])

    # pocket_file
    pocket_file = os.path.join(args.protein_pockets_dir, f"{proteinName}_{pocketName}.pdb")

    # temporary ligand smi path
    compound_info = []
    for ligand in ['DB16400.sdf', 'DB05129.sdf']:
        ligandName = ligand.split('.sdf')[0]
        ligandFile = os.path.join(args.ligandDirs, ligand)
        mol, error = read_mol(ligandFile, None)  # unreadable by rdkit
        if error:
            continue
        if mol.GetNumAtoms() < 2: # single atom?
            continue

        smiles = Chem.MolToSmiles(mol)
        compound_info.append((ligandName, smiles))

        with open('./temp_ligand.smi', 'w') as f:
            for compound in compound_info:
                f.write(compound[0]+' '+compound[1]+'\n')

    num_trails = 5
    for trail in range(1, num_trails + 1):
        try:
            test_dataset = VSTestGraphDataset_Fly_SMI(protein_file=pocket_file,
                                                      ligand_path='./temp_ligand.smi',
                                                      pocket_center=pocket_center,
                                                      protein_name=proteinName)
        except EOFError as e:
            print(e, proteinName)
            exit()

        test_dataset.generate_graphs(ligand_smi='./temp_ligand.smi', n_job=-1)  # use CPU to generate graph -> this may take time
        protein_name_pocket, compound_name_pocket, corrected_binding_scores_pocket, pocket_names_pocket = make_inference(
                test_dataset=test_dataset,
                model=model,
                device=device,
                proteinname=proteinName,
                pocketname=pocketName,
                out_dir=os.path.join(args.out_vis_dir, proteinName),
                postprocessing=True
        )
        print(compound_name_pocket, corrected_binding_scores_pocket)



    '''Filter results'''
    # for proteinName in os.listdir(args.proteinDirsUni):
    #     all_trials_data = pd.DataFrame()
    #
    #     for trail in range(1, 6):
    #         results_csv = f"{args.out_dir}/protein5_to_drugbank9k_{proteinName}_results_trail{trail}.csv"
    #         df_score = pd.read_csv(results_csv)
    #         all_trials_data = all_trials_data.append(df_score)
    #
    #     average_df_score = all_trials_data.groupby(['protein_name', 'compound_name', 'pocket_name', 'pocket_com'])['binding_strength'].mean().reset_index()
    #
    #     average_df_score = average_df_score.sort_values(by='binding_strength', ascending=False)  # sort from the highest to the lowest prediction confidences
    #     average_df_score = average_df_score.loc[average_df_score.groupby('compound_name')['binding_strength'].transform(max) == average_df_score['binding_strength']] # for each compound, get its highest binding strength
    #     average_df_score.to_csv(f"{args.out_dir}/protein5_to_drugbank9k_{proteinName}_results_average.csv", index=False)
    #
    #     chosen = average_df_score.query(f"binding_strength > {args.score_threshold}").reset_index(drop=True)
    #     chosen = chosen.sort_values(by='binding_strength', ascending=False)  # sort from the highest to the lowest prediction confidences
    #
    #     strength_list = list(average_df_score['binding_strength'])
#
#         fig = plt.figure(figsize=(10, 6))
#         gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
#         ax0 = plt.subplot(gs[0])
#         count, bins, ignored = ax0.hist(strength_list, bins=30, density=True,
#                                         alpha=0.6, color='skyblue',
#                                         edgecolor='black')
#         # Set the x-axis limits to align both plots
#         xmin = min(strength_list) - 1
#         xmax = max(strength_list) + 1
#         ax0.set_xlim([xmin, xmax])
#
#         # Add title for the histogram
#         ax0.set_title(f'Binding strength distribution for Protein {proteinName}')
#
#         # Create the box plot
#         ax1 = plt.subplot(gs[1], sharex=ax0)
#         ax1.boxplot(strength_list, vert=False, widths=0.6)
#         ax1.grid(True, which='major', linestyle='--', linewidth=0.5)
#         ax1.set_yticks([])
#
#         # Annotate the threshold cutoff and max value
#         threshold = args.score_threshold
#         max_value = max(strength_list)
#
#         # ax0.annotate(f'Threshold = {args.score_threshold}',
#         #              xy=(threshold, 0),
#         #              xytext=(threshold, 0.02),
#         #              arrowprops=dict(facecolor='black', shrink=0.05, zorder=10),
#         #              horizontalalignment='center', fontweight='bold', clip_on=False, zorder=10)
#
#         # Max value annotation
#         ax0.annotate(f'Max value: {max_value:.2f}', xy=(max_value, 0), xytext=(max_value - 10, 0.02),
#                      arrowprops=dict(facecolor='black', shrink=0.05, zorder=10),
#                      horizontalalignment='left', fontweight='bold', clip_on=False, zorder=10)
#
#         # Set the labels for the box plot
#         ax1.set_xlabel('Value')
#
#         # Adjust layout to fit the annotation text
#         plt.tight_layout()
#
#         # Show the plot
#         plt.show()
#         print()
# #
#