import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk
from collections import defaultdict
import hdbscan
from sklearn.datasets import make_blobs





def import_data():    # Import data from CSV files
    player_stats = pd.read_csv('player-data/all_players_and_stats_2024-25.csv')

    '''print(f"player_stats shape: {player_stats.shape}")
    print(f"player_stats first 5 rows:\n{player_stats.head(5)}")'''

    return player_stats


def find_player_stats(player_stats):
    # There are 2309 total players in the dataset
    num_of_players = 2309
    #                      0     1        2        3       4         5          6         7              8             9      10.      11.      12       13     14      15      16      17      18     19     20    21    22     23    24     25      26     27    28      29          30         31           32             33              34           35          36.     37         38        39.      40.        41.       42
    column_list_names = ["RK", "Pick", "Year", "Height", "Name", "Unknown1", "Team", "Conference", "Games Played", "Role", "MIN%", "PRPG!", "D-PRPG", "BPM", "OBPM", "DBPM", "ORTG", "D-RTG", "USG", "EFG", "TS", "OR", "DR", "AST", "TO", "A/TO", "BLK", "STL", "FTR", "FC/40", "Dunk Ratio", "Dunk %", "Close 2 Ratio", "Close 2 %", "Far 2 Ratio", "Far 2 %", "FT Ratio", "FT %", "2P Ratio", "2P %", "3P/100", "3P Ratio", "3P %"]
    
    # Parameters for stats to compare
    
    # Things to add:
    # 1. Display good/bad outlier players for the data set
    # 2. Display percentiles of stats for each cluster
    stats = [11,12,13,14,15]

    stat_1 = stats[0]     ##
    stat_2 = stats[1]     ##

    n_clusters = 1
    eps = 0.7
    min_samples = 5
    min_cluster_size = 2    # For HDBSCAN

    stat_names = []

    for stat in stats:
        stat_name = column_list_names[stat]
        stat_names.append(stat_name)

    stat_1_name = stat_names[0]    ##
    stat_2_name = stat_names[1]     ##
    # print(f"player_stats columns names: \n {column_list_names}\n")
    
    # Would like to find a way to filter out players with certain conditions, like conference, team, role, etc.
    print(f"type of player_stats.iloc[0, 7]: {type(player_stats.iloc[0, 7])}")

    X_list = []
    y_list = []
    all_stats = [[] for _ in stats]   ##
    name_list = []



    for i in range(num_of_players-1):
        player_team = player_stats.iloc[i, 6]
        player_conference = player_stats.iloc[i, 7]

        if player_conference == "A10":
            # name_counter = 0
            for j, stat in enumerate(stats):
                all_stats[j].append(player_stats.iloc[i, stat])

            X_list.append(player_stats.iloc[i, stat_1])   ##
            y_list.append(player_stats.iloc[i, stat_2])   ##
            name_list.append(player_stats.iloc[i, 4])




    if pd.Series(X_list).isnull().any():
        print(f"Warning: {stat_1_name} contains null values. Please check the data.")
        return None, None, None, None, None, n_clusters
    


    return X_list, y_list, all_stats, name_list, stat_1_name, stat_2_name, stat_names, n_clusters, eps, min_samples, min_cluster_size



########################################################################################################################################################################################


def get_player_selection(name_list):
    """Create a GUI window for player selection with dropdown"""
    selected_player = None
    selected_index = None
    
    def on_select():
        nonlocal selected_player, selected_index
        current_text = search_var.get().strip()
        if current_text:
            # Find exact match in the filtered list
            for i, name in enumerate(name_list):
                if str(name).strip() == current_text:
                    selected_player = name
                    selected_index = i
                    break
        root.quit()
    
    def on_skip():
        nonlocal selected_player, selected_index
        selected_player = None
        selected_index = None
        root.quit()
    
    def update_dropdown(*args):
        """Update dropdown options based on search text"""
        search_text = search_var.get().lower()
        if len(search_text) >= 2:  # Start filtering after 2 characters
            filtered_names = [str(name) for name in name_list if search_text in str(name).lower()]
            search_combo['values'] = filtered_names[:20]  # Limit to 20 results
        else:
            search_combo['values'] = []
    
    # Create main window
    root = tk.Tk()
    root.title("Player Search")
    root.geometry("400x200")
    root.resizable(False, False)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    # Create and pack widgets
    title_label = tk.Label(root, text="Search for a Player to Highlight", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)
    
    instruction_label = tk.Label(root, text="Type a player name (minimum 2 characters):")
    instruction_label.pack(pady=5)
    
    # Search variable and combobox
    search_var = tk.StringVar()
    search_var.trace('w', update_dropdown)
    
    search_combo = ttk.Combobox(root, textvariable=search_var, width=40)
    search_combo.pack(pady=10)
    
    # Button frame
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    select_button = tk.Button(button_frame, text="Highlight Player", command=on_select, 
                             bg="lightgreen", font=("Arial", 10, "bold"))
    select_button.pack(side=tk.LEFT, padx=10)
    
    skip_button = tk.Button(button_frame, text="Skip", command=on_skip,
                           bg="lightgray", font=("Arial", 10))
    skip_button.pack(side=tk.LEFT, padx=10)
    
    # Focus on the search box
    search_combo.focus()
    
    # Start the GUI
    root.mainloop()
    root.destroy()
    
    return selected_player, selected_index

############################################################################################################################################################################

def plot_basic_stats(X_list, y_list, name_list, X_name, y_name):
    
    # Use GUI to get player selection instead of command line
    highlighted_player, highlighted_index = get_player_selection(name_list)
    
    # Create colors array - default blue, red for searched player
    colors = ['blue'] * len(X_list)
    
    if highlighted_index is not None:
        colors[highlighted_index] = 'red'
        print(f"Highlighted player: {highlighted_player}")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_list, y_list, c=colors, alpha=0.7)
    ax.set_title(f'{X_name} vs {y_name} for Players')
    ax.set_xlabel(f'{X_name}')
    ax.set_ylabel(f'{y_name}')
    
    # Add legend if player is highlighted
    if highlighted_player:
        ax.text(0.02, 0.98, f"Highlighted: {highlighted_player}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8))

    # Create hover functionality
    annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="lightblue", alpha=1.0),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = f"{name_list.iloc[ind['ind'][0]]}"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('lightblue')
        annot.get_bbox_patch().set_alpha(1.0)  # Solid background (no transparency)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

#****************************************************************************************************************************************************************************************


def find_kmeans(X_list, y_list, name_list, X_name, y_name, n_clusters=5):
    # 2. Apply K-Means
    kmeans_input = np.column_stack((X_list, y_list))  # shape (n_samples, 2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # specify number of clusters
    kmeans.fit(kmeans_input)

    # 3. Get cluster assignments and centroids
    y_kmeans = kmeans.predict(kmeans_input)
    centers = kmeans.cluster_centers_

    # 4. Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=y_kmeans, s=30, cmap='viridis')
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X')
    ax.set_title("K-Means Clustering --- " + X_name + " vs " + y_name)
    ax.set_xlabel(f'{X_name}')
    ax.set_ylabel(f'{y_name}')

    # Create hover functionality
    annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="lightblue", alpha=1.0),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = f"{name_list[ind['ind'][0]]}"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('lightblue')
        annot.get_bbox_patch().set_alpha(1.0)  # Solid background (no transparency)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

#****************************************************************************************************************************************************************************************


def find_dbscan(list_of_stats, name_list, list_of_stat_names, EPS=0.5, MIN_SAMPLES=5):

    print(f"list_of_stats[0] = {list_of_stats[0]}")

    # IF WE HAVE TIME, LOOK AT FINDING OUT WHY NAMES ARE NOT POPPING UP WHEN THE MOUSE HOVERS OVER DOTS

    # Standardize the features
    N_COMPONENTS = len(list_of_stats)
    print(f"N_COMPONENTS: {N_COMPONENTS}")
    original_X = np.array(list_of_stats).T
    X = StandardScaler().fit_transform(np.array(list_of_stats).T)

    # Apply PCA for dimensionality reduction
    # PCA = Principle Component Analysis
    pca = PCA(n_components=N_COMPONENTS)
    X_pca = pca.fit_transform(X)

    X_pca = X

    # Apply DBSCAN ###############################################################################################
    clusterer = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    cluster_labels = clusterer.fit_predict(X_pca)

    # Group player names by their cluster labels
    cluster_groups_names = defaultdict(list)
    # Holds all stats for players in ALL clusters
    all_cluster_stats = defaultdict(list)
    average_cluster_stats = defaultdict(list)

    print(f"original_X shape: {original_X.shape}, type: {type(original_X)}")
    # print(f"original_X = {original_X}")
    # return

    for idx, label in enumerate(cluster_labels):

        player_stat_list = original_X[idx]
        player_name = name_list[idx]
        cluster_groups_names[label].append(player_name)
        all_cluster_stats[label].append(player_stat_list)
        # Find the average stats for each cluster

        '''for idx, item in enumerate(player_stat_list):
            stat_name = list_of_stat_names[idx]
            all_cluster_stats[stat_name].append(item)'''

        # all_cluster_stats[label].append(player_stat_list)
    # Loops through each cluster label
    for cluster_label, cluster_stat_list in all_cluster_stats.items():
        # Holds all stats for players in a SINGLE cluster, labels are stat names
        accumulated_single_cluster_stats = defaultdict(list)
        
        # Loops through each player's stats in that cluster
        for player_idx, player_stat_list in enumerate(cluster_stat_list):

            # Loops through each stat of that player
            for stat_idx, stat in enumerate(player_stat_list):
                stat_name = list_of_stat_names[stat_idx]
                accumulated_single_cluster_stats[stat_name].append(stat)
                # print(f"Cluster {label}, Player {cluster_groups_names[label][idx]}, Stats: {stat}, type of {type(stat)}, type(stat[0]): {type(stat[0])}, stat_name: {stat_name}")


        # Now find the average of each stat for that cluster
        average_cluster_stats[cluster_label] = []
        for stat_name, stats in accumulated_single_cluster_stats.items():
            average_stat = np.mean(stats)
            average_cluster_stats[cluster_label].append(average_stat)

            # print(f"Cluster {label}, Player {cluster_groups_names[label][idx]}, Stats: {stat}, type of {type(stat)}, type(stat[0]): {type(stat[0])}, stat_name: {stat_name}")

    print(f"Cluster Groups and their Players:\n")
    for label, players in cluster_groups_names.items():
        print(f"  Cluster {label}: {players}\n")
    print(f"Average Stats per Cluster:\n\n")
    temp_idx = 0
    for label, avg_stats in average_cluster_stats.items():
        print(f"  Cluster {label}:")
        print(f"    Number of Players: {len(cluster_groups_names[label])}")
        for stat_name, avg_stat in zip(list_of_stat_names, avg_stats):
            print(f"    {stat_name}: {avg_stat:.2f}")
        print(f"        Sum of all stats: {sum(avg_stats):.2f}")
        print()
        temp_idx += 1
        '''if temp_idx >= 5:
            break'''


    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(original_X[:, 0], original_X[:, 1], c=cluster_labels, s=30, cmap='jet')
    ax.set_title("DBSCAN Clustering")
    ax.set_xlabel(f'{list_of_stat_names[0]}')
    ax.set_ylabel(f'{list_of_stat_names[1]}')

    # Create hover functionality
    annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="lightblue", alpha=1.0),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)


    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = f"{name_list[ind['ind'][0]]}"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('lightblue')
        annot.get_bbox_patch().set_alpha(1.0)  # Solid background (no transparency)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


#****************************************************************************************************************************************************************************************

# This is a copy and paste of find_dbscan, with some modifications to change it to hdbscan
def find_hdbscan(list_of_stats, name_list, list_of_stat_names, MIN_CLUSTER_SIZE=5, MIN_SAMPLES=5):
    # Idk if I will need min_samples for hdbscan, but I am including it for now

    print(f"list_of_stats[0] = {list_of_stats[0]}")

    # IF WE HAVE TIME, LOOK AT FINDING OUT WHY NAMES ARE NOT POPPING UP WHEN THE MOUSE HOVERS OVER DOTS

    # Standardize the features
    N_COMPONENTS = len(list_of_stats)
    print(f"N_COMPONENTS: {N_COMPONENTS}")
    original_X = np.array(list_of_stats).T
    X = StandardScaler().fit_transform(np.array(list_of_stats).T)

    # Apply PCA for dimensionality reduction
    # PCA = Principle Component Analysis
    pca = PCA(n_components=N_COMPONENTS)
    X_pca = pca.fit_transform(X)

    X_pca = X

    # Apply HDBSCAN ################################################################################
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE)
    cluster_labels = clusterer.fit_predict(X_pca)

    # Group player names by their cluster labels
    cluster_groups_names = defaultdict(list)
    # Holds all stats for players in ALL clusters
    all_cluster_stats = defaultdict(list)
    average_cluster_stats = defaultdict(list)

    print(f"original_X shape: {original_X.shape}, type: {type(original_X)}")
    # print(f"original_X = {original_X}")
    # return

    for idx, label in enumerate(cluster_labels):

        player_stat_list = original_X[idx]
        player_name = name_list[idx]
        cluster_groups_names[label].append(player_name)
        all_cluster_stats[label].append(player_stat_list)
        # Find the average stats for each cluster

        '''for idx, item in enumerate(player_stat_list):
            stat_name = list_of_stat_names[idx]
            all_cluster_stats[stat_name].append(item)'''

        # all_cluster_stats[label].append(player_stat_list)
    # Loops through each cluster label
    for cluster_label, cluster_stat_list in all_cluster_stats.items():
        # Holds all stats for players in a SINGLE cluster, labels are stat names
        accumulated_single_cluster_stats = defaultdict(list)
        
        # Loops through each player's stats in that cluster
        for player_idx, player_stat_list in enumerate(cluster_stat_list):

            # Loops through each stat of that player
            for stat_idx, stat in enumerate(player_stat_list):
                stat_name = list_of_stat_names[stat_idx]
                accumulated_single_cluster_stats[stat_name].append(stat)
                # print(f"Cluster {label}, Player {cluster_groups_names[label][idx]}, Stats: {stat}, type of {type(stat)}, type(stat[0]): {type(stat[0])}, stat_name: {stat_name}")


        # Now find the average of each stat for that cluster
        average_cluster_stats[cluster_label] = []
        for stat_name, stats in accumulated_single_cluster_stats.items():
            average_stat = np.mean(stats)
            average_cluster_stats[cluster_label].append(average_stat)

            # print(f"Cluster {label}, Player {cluster_groups_names[label][idx]}, Stats: {stat}, type of {type(stat)}, type(stat[0]): {type(stat[0])}, stat_name: {stat_name}")

    print(f"Cluster Groups and their Players:\n")
    for label, players in cluster_groups_names.items():
        print(f"  Cluster {label}: {players}\n")
    print(f"Average Stats per Cluster:\n\n")
    temp_idx = 0
    for label, avg_stats in average_cluster_stats.items():
        print(f"  Cluster {label}:")
        print(f"    Number of Players: {len(cluster_groups_names[label])}")
        for stat_name, avg_stat in zip(list_of_stat_names, avg_stats):
            print(f"    {stat_name}: {avg_stat:.2f}")
        print(f"        Sum of all stats: {sum(avg_stats):.2f}")
        print()
        temp_idx += 1
        '''if temp_idx >= 5:
            break'''


    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(original_X[:, 0], original_X[:, 1], c=cluster_labels, s=30, cmap='jet')
    ax.set_title("HDBSCAN Clustering")
    ax.set_xlabel(f'{list_of_stat_names[0]}')
    ax.set_ylabel(f'{list_of_stat_names[1]}')

    # Create hover functionality
    annot = ax.annotate('', xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="lightblue", alpha=1.0),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)


    def update_annot(ind):
        pos = scatter.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = f"{name_list[ind['ind'][0]]}"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('lightblue')
        annot.get_bbox_patch().set_alpha(1.0)  # Solid background (no transparency)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()


def main():

    # TO DO:
    # 1. Import data
    # 2. Figure out how to classify data
    # 2A - We can use clustering to classify players based on their stats. We can use K-Means or Hierarchical Clustering. We can also chart a 2d graph with basic stats. 
    # 3. Save data somewhere

    player_stats = import_data()

    # X_list and y_list are the first two stats in all_stats
    # X_name and y_name are the names of those stats

    X_list, y_list, all_stats, name_list, X_name, y_name, stat_names, n_clusters, eps, min_samples, min_cluster_size = find_player_stats(player_stats)

    print(f"player_stats info: \n {player_stats[0:5]}\n")

    print(f"player_stats columns: \n {player_stats.columns}\n")

    print(f"player_stats players iloc: \n {player_stats.iloc[0:5, 0:5]}")

    # plot_basic_stats(X_list, y_list, name_list, X_name, y_name)

    # find_kmeans(X_list, y_list, name_list, X_name, y_name, n_clusters)

    #find_dbscan(all_stats, name_list, stat_names, eps, min_samples)

    find_hdbscan(all_stats, name_list, stat_names, min_cluster_size, min_samples)

    pass

if __name__ == "__main__":
    main()