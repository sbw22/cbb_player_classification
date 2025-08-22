import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import tkinter as tk
from tkinter import ttk





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
    stat_1 = 33
    stat_2 = 35
    n_clusters = 1

    stat_1_name = column_list_names[stat_1]
    stat_2_name = column_list_names[stat_2]
    # print(f"player_stats columns names: \n {column_list_names}\n")
    

    name_list = player_stats.iloc[0:num_of_players, 4]
    X_list = player_stats.iloc[0:num_of_players, stat_1]
    y_list = player_stats.iloc[0:num_of_players, stat_2]

    return X_list, y_list, name_list, stat_1_name, stat_2_name, n_clusters



############################################################################################################################################################################


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



def find_kmeans(X_list, y_list, name_list, X_name, y_name, n_clusters=5):
    # 2. Apply K-Means
    kmeans_input = np.column_stack((X_list, y_list))  # shape (n_samples, 2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # specify number of clusters
    kmeans.fit(kmeans_input)

    # 3. Get cluster assignments and centroids
    y_kmeans = kmeans.predict(kmeans_input)
    centers = kmeans.cluster_centers_

    # 4. Plot the results
    plt.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=y_kmeans, s=30, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X')
    plt.title("K-Means Clustering --- " + X_name + " vs " + y_name)
    plt.xlabel(f'{X_name}')
    plt.ylabel(f'{y_name}')
    plt.show()



def main():

    # TO DO:
    # 1. Import data
    # 2. Figure out how to classify data
    # 2A - We can use clustering to classify players based on their stats. We can use K-Means or Hierarchical Clustering. We can also chart a 2d graph with basic stats. 
    # 3. Save data somewhere

    player_stats = import_data()

    X_list, y_list, name_list, X_name, y_name, n_clusters = find_player_stats(player_stats)

    print(f"player_stats info: \n {player_stats[0:5]}\n")

    print(f"player_stats columns: \n {player_stats.columns}\n")

    print(f"player_stats players iloc: \n {player_stats.iloc[0:5, 0:5]}")

    # plot_basic_stats(X_list, y_list, name_list, X_name, y_name)

    find_kmeans(X_list, y_list, name_list, X_name, y_name, n_clusters)

    pass

if __name__ == "__main__":
    main()