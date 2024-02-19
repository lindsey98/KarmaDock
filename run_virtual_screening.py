
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
from utils.post_processing import correct_pos
from utils.pre_processing import get_pocket_pure, read_mol
import rdkit.Chem as Chem
from Bio.PDB import PDBParser
import glob
import re



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


@torch.no_grad()
def make_inference(test_dataset, model, device, proteinname, pocketname, postprocessing=False):
    protein_name = []
    compound_name = []
    binding_scores = []
    pocket_names = []

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
            lig_pos, mdn_score_pred = post_processing(lig_node_s=lig_node_s,
                                                      pro_node_s=pro_node_s,
                                                      lig_pos=lig_pos,
                                                      data=data,
                                                      mdn_score_pred=mdn_score_pred)

        protein_name.extend([proteinname] * batch_size)
        compound_name.extend(data.pdb_id)
        pocket_names.extend([pocketname] * batch_size)
        # binding_scores.extend(mdn_score_pred.cpu().numpy().tolist())
        binding_scores.extend([round(score, 4) for score in mdn_score_pred.cpu().numpy().tolist()])

    return protein_name, compound_name, binding_scores, pocket_names

@torch.no_grad()
def post_processing(lig_node_s, pro_node_s, lig_pos, data, mdn_score_pred, use_ff=True):
    # # post processing
    data.pos_preds = lig_pos
    poses, _, _ = correct_pos(data,
                              mask=mdn_score_pred <= args.score_threshold,
                              out_dir=args.out_dir,
                              out_init=args.out_init,
                              out_uncoorected=args.out_uncoorected,
                              out_corrected=args.out_corrected)

    if use_ff:
        ff_corrected_pos = torch.from_numpy(np.concatenate([i[0] for i in poses], axis=0)).to(args.device)
        mdn_score_pred_ff_corrected = model.module.scoring(lig_s=lig_node_s,
                                                           lig_pos=ff_corrected_pos,
                                                           pro_s=pro_node_s,
                                                           data=data,
                                                           dist_threhold=5.,
                                                           batch_size=args.batch_size)
        return ff_corrected_pos, mdn_score_pred_ff_corrected

    else:
        align_corrected_pos = torch.from_numpy(np.concatenate([i[1] for i in poses], axis=0)).to(args.device)
        mdn_score_pred_align_corrected = model.module.scoring(lig_s=lig_node_s,
                                                              lig_pos=align_corrected_pos,
                                                              pro_s=pro_node_s,
                                                              data=data,
                                                              dist_threhold=5.,
                                                              batch_size=args.batch_size)
        return align_corrected_pos, mdn_score_pred_align_corrected

if __name__ == '__main__':

    # get parameters from command line
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--out_dir', type=str,
                           default='./datasets/protein315_to_drugbank9k_results_csv',
                           help='dir for recording binding poses and binding scores')

    argparser.add_argument('--score_threshold', type=float,
                           default=50,
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
    argparser.add_argument('--proteinDirsUni', default="/home/ruofan/git_space/TankBind/datasets/protein_315", help='Directory for protein files (unfied)')

    argparser.add_argument('--protein_pockets_p2rank', default="/home/ruofan/git_space/TankBind/datasets/protein_315_p2rank",
                        help='Where to save the Protein pockets p2rank predictions')
    argparser.add_argument('--ligandDirs', default="/home/ruofan/git_space/TankBind/datasets/drugbank", help='Directory for ligand files')

    argparser.add_argument('--protein_pockets_dir', type=str,
                           default='./datasets/protein_315_p2rank_processedPDB',
                           help='the protein files path')

    argparser.add_argument('--ligand_smi', type=str,
                           default='./datasets/drugbank_9k.smi',
                           help='Where to save the ligands smile')

    argparser.add_argument('--modelFile', default="./trained_models/karmadock_screening.pkl", help='Pretrained model file path')

    argparser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')


    args = argparser.parse_args()
    set_random_seed(args.random_seed)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.protein_pockets_dir, exist_ok=True)

    protein_dict = torch.load('/home/ruofan/git_space/TankBind/datasets/protein_315.pt') # this is processed by TankBind

    '''Aggregate all predicted pocket centers (as well as the protein center) as potential binding sites, and expand a radius of 20A as the potential binding pockets'''
    for proteinName in list(protein_dict.keys()):

        ### read the p2rank prediction results, get all predicted pocket centers
        p2rankFile = f"{args.protein_pockets_p2rank}/{proteinName}.pdb_predictions.csv"
        pocket = pd.read_csv(p2rankFile)
        pocket.columns = pocket.columns.str.strip()
        pocket_coms = pocket[['center_x', 'center_y', 'center_z']].values # get all predicted pocket centers

        ### get the protein center as a pocket as well
        protein_pdb = os.path.join(args.proteinDirsUni, proteinName + ".pdb")
        protein_center = get_calpha_center_coords(protein_pdb)

        ### select the corresponding pocket according to the pocket center, use radius=20 as TankBind does
        pocket_file = os.path.join(args.protein_pockets_dir, proteinName+'_pocket_center.pdb')
        if not os.path.exists(pocket_file):
             get_pocket_pure(os.path.join(args.proteinDirsUni, proteinName+".pdb"),
                            somepoint=protein_center,
                            out_file=pocket_file,
                            size=20)

        for i, com in enumerate(pocket_coms):
            pocket_file = os.path.join(args.protein_pockets_dir, proteinName + f'_pocket_{i+1}.pdb')
            if not os.path.exists(pocket_file):
                get_pocket_pure(protein_pdb,
                                somepoint=np.asarray([com]),
                                out_file=pocket_file,
                                size=20)

    '''Write all ligands in a single smile file'''
    if (not os.path.exists(args.ligand_smi)) or len(open(args.ligand_smi).read()) == 0:
        compound_info = []
        for ligand in tqdm(os.listdir(args.ligandDirs)):
            ligandName = ligand.split('.sdf')[0]
            ligandFile = os.path.join(args.ligandDirs, ligand)
            mol, error = read_mol(ligandFile, None)  # unreadable by rdkit
            if error:
                continue
            if mol.GetNumAtoms() < 2: # single atom?
                continue

            smiles = Chem.MolToSmiles(mol)
            compound_info.append((ligandName, smiles))

        with open(args.ligand_smi, 'w') as f:
            for compound in compound_info:
                f.write(compound[0]+' '+compound[1]+'\n')

    '''Make predictions'''
    # # load model
    device = args.device
    model = KarmaDock()
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(device)
    model.load_state_dict(torch.load(args.modelFile, map_location=device)['model_state_dict'],
                          strict=False)
    model.eval()

    for proteinName in list(protein_dict.keys())[::-1]:
        if os.path.exists(f"{args.out_dir}/protein315_to_drugbank9k_{proteinName}_results.csv"):
            continue

        # time
        protein_name = []
        compound_name = []
        binding_scores = []
        pocket_names = []
        pocket_coms = []

        ### read the p2rank prediction results, get all predicted pocket centers
        p2rankFile = f"{args.protein_pockets_p2rank}/{proteinName}.pdb_predictions.csv"
        pocket = pd.read_csv(p2rankFile)
        pocket.columns = pocket.columns.str.strip()
        pocket_coords = pocket[['center_x', 'center_y', 'center_z']].values # get all predicted pocket centers

        search_pattern = os.path.join(args.protein_pockets_dir, f"{proteinName}_pocket_*.pdb")
        pocket_files = glob.glob(search_pattern)
        print(f"Found {len(pocket_files)} pockets for {proteinName}.")

        for i, pocket_file in enumerate(pocket_files):
            pattern = r"/([^/]+)_(pocket_(?:center|\d+))\.pdb$"
            match = re.search(pattern, pocket_file)
            pocketName = match.group(2)  #

            if pocketName.split('pocket_')[1] == 'center':
                pocket_center = torch.Tensor([a for a in protein_dict[proteinName][0].mean(axis=0).numpy()])
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
                                                                                                                  pocketname=pocketName)
            protein_name.extend(protein_name_pocket)
            compound_name.extend(compound_name_pocket)
            binding_scores.extend(binding_scores_pocket)
            pocket_names.extend(pocket_names_pocket)
            pocket_coms.extend([",".join([str(a.round(3)) for a in pocket_center.numpy()])]*len(protein_name_pocket))


        # out to csv
        df_score = pd.DataFrame(list(zip(protein_name, compound_name, pocket_names, pocket_coms, binding_scores)),
                                columns=['protein_name', 'compound_name', 'pocket_name', 'pocket_com', 'binding_strength'])

        df_score.to_csv(f"{args.out_dir}/protein315_to_drugbank9k_{proteinName}_results.csv", index=False)

    '''Filter results'''
    # for proteinName in list(protein_dict.keys())[::-1]:
    #     results_csv = f"{args.out_dir}/protein315_to_drugbank9k_{proteinName}_results.csv"
    #     df_score = pd.read_csv(results_csv)
    #     df_score['original_index'] = df_score.index
    #     top3 = df_score.groupby('protein_name', group_keys=False).apply(lambda x: x.nlargest(3, 'binding_strength')) # get top-3 results
    #     chosen = top3.query(f"binding_strength > {args.score_threshold}").reset_index(drop=True)
    #     print(chosen)
