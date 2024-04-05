import os
import subprocess
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    parent_dir = "/home/ruofan/git_space/KarmaDock/DEKOIS2/"
    script_dir = "/home/ruofan/git_space/KarmaDock/utils"
    conda_python_path = "/home/ruofan/anaconda3/envs/karmadock/bin/python"

    # Store the original directory to revert back later
    original_dir = os.getcwd()

    # Change to the directory where the script is located
    os.chdir(script_dir)

    # for entry in tqdm(os.listdir(parent_dir)):
    #     subdir_path = os.path.join(parent_dir, entry)
    #     if os.path.exists(f"{subdir_path}/karmadocked") and len(os.listdir(f"{subdir_path}/karmadocked"))>0:
    #         continue
    #     # Check if this entry is a directory
    #     if os.path.isdir(subdir_path):
    #         try:
    #             # Construct and execute the first command
    #             cmd1 = f"{conda_python_path} -u virtual_screening_pipeline.py " \
    #                    f"--ligand_smi {subdir_path}/active_decoys.smi " \
    #                    f"--protein_file {subdir_path}/protein/{entry}_protein.pdb " \
    #                    f"--crystal_ligand_file {subdir_path}/protein/{entry}_ligand.mol2 " \
    #                    f"--graph_dir {subdir_path}/karmadock_liggraph " \
    #                    f"--out_dir {subdir_path}/karmadocked " \
    #                    f"--random_seed 2023 " \
    #                    f"--mode generate_graph"
    #             subprocess.run(cmd1, shell=True, check=True)
    #
    #             # Construct and execute the second command
    #             cmd2 = f"{conda_python_path} -u virtual_screening_pipeline.py " \
    #                    f"--protein_file {subdir_path}/protein/{entry}_protein.pdb " \
    #                    f"--crystal_ligand_file {subdir_path}/protein/{entry}_ligand.mol2 " \
    #                    f"--graph_dir {subdir_path}/karmadock_liggraph " \
    #                    f"--out_dir {subdir_path}/karmadocked " \
    #                    f"--score_threshold 50 " \
    #                    f"--batch_size 64 " \
    #                    f"--random_seed 2023 " \
    #                    f"--out_uncoorected " \
    #                    f"--out_corrected"
    #             subprocess.run(cmd2, shell=True, check=True)
    #         except subprocess.CalledProcessError:
    #             continue
    #
    # # Revert back to the original directory
    # os.chdir(original_dir)

    score_collection = {}
    all_active_scores = []
    all_decoy_scores = []
    for entry in tqdm(os.listdir(parent_dir)):
        subdir_path = os.path.join(parent_dir, entry)
        score_csv = f"{subdir_path}/karmadocked/score.csv"
        if os.path.exists(score_csv):
        #     df = pd.read_csv(score_csv)
        #     df['category'] = df['pdb_id'].apply(lambda x: 'ZINC' if x.startswith('ZINC') else ('BDB' if x.startswith('BDB') else 'Other'))
        #
        #     # Count the occurrences of each category
        #     category_counts = df['category'].value_counts()
        #     # Identify which category is 'decoy' and which is 'active'
        #     decoy = category_counts.idxmax()
        #     active = category_counts.idxmin()
        #
        #     # Map the 'decoy' and 'active' labels back to the DataFrame
        #     df['type'] = df['category'].apply(lambda x: 'decoy' if x == decoy else 'active')
        #     df.to_csv(score_csv, index=False)

            df = pd.read_csv(score_csv)
            active_scores = df[df['type'] == 'active']['karma_score'].tolist()
            decoy_scores = df[df['type'] == 'decoy']['karma_score'].tolist()
            score_collection[entry] = {'active': active_scores,
                                       'decoy': decoy_scores}

            all_active_scores.extend(active_scores)
            all_decoy_scores.extend(decoy_scores)

    # # Creating a DataFrame
    # plotdf = pd.DataFrame({'Scores': all_active_scores + all_decoy_scores,
    #                        'KarmaDock Score for different types of compounds': ['Actives'] * len(all_active_scores) + ['Decoys'] * len(all_decoy_scores)})
    #
    # # Plotting
    # sns.boxplot(x='KarmaDock Score for different types of compounds', y='Scores', data=plotdf)
    # plt.title('Boxplot for Two Lists')
    # plt.show()

    # Generating a range of possible thresholds
    thresholds = np.linspace(min(all_decoy_scores + all_active_scores), max(all_decoy_scores + all_active_scores), 100)

    # Calculate classification accuracy for each threshold
    accuracies = []
    for threshold in thresholds:
        true_positives = sum(score >= threshold for score in all_active_scores)
        true_negatives = sum(score < threshold for score in all_decoy_scores)
        accuracy = (true_positives + true_negatives) / (len(all_active_scores) + len(all_decoy_scores))
        accuracies.append(accuracy)

    # identify the elbow
    norm_thresholds = (thresholds - min(thresholds)) / (max(thresholds) - min(thresholds))
    norm_accuracies = (accuracies - min(accuracies)) / (max(accuracies) - min(accuracies))
    start_point = np.array([norm_thresholds[0], norm_accuracies[0]])
    end_point = np.array([norm_thresholds[-1], norm_accuracies[-1]])
    line_vec = end_point - start_point
    distances = []
    for (x, y) in zip(norm_thresholds, norm_accuracies):
        point_vec = np.array([x, y]) - start_point
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec * 1.0 / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        nearest = line_unitvec * t
        dist = np.linalg.norm(point_vec_scaled - nearest)
        distances.append(dist)

    # Identify the elbow point
    elbow_index = np.argmax(distances)
    elbow_threshold = thresholds[elbow_index]
    elbow_accuracy = accuracies[elbow_index]

    # Plotting with elbow marked
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies, marker='o', linestyle='-', markersize=5)
    plt.scatter(elbow_threshold, elbow_accuracy, color='red', s=100, label='Elbow Point', zorder=5)
    plt.xlabel('Threshold')
    plt.ylabel('Classification Accuracy (Actives vs. Decoys)')
    plt.legend()
    plt.grid(True)
    plt.axvline(x=elbow_threshold, color='r', linestyle='--',
                label=elbow_threshold)  # Dashed vertical line
    plt.text(elbow_threshold, -0.05, f'{elbow_threshold:.2f}', ha='center', va='bottom', color='red')
    plt.show()

