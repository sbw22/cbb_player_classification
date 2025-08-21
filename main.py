import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs



def import_data():    # Import data from CSV files
    player_stats = pd.read_csv('player-data/all_players_and_stats_2024-25.csv')

    '''print(f"player_stats shape: {player_stats.shape}")
    print(f"player_stats first 5 rows:\n{player_stats.head(5)}")'''

    return player_stats



def plot_basic_stats(player_stats):

    # There are 2309 total players in the dataset
    num_of_players = 2309
    #                      0     1        2        3       4         5          6         7              8             9      10.      11.      12       13     14      15      16      17      18     19     20    21    22     23    24     25      26     27    28      29          30         31           32             33              34           35          36.     37         38        39.      40.        41.       42
    column_list_names = ["RK", "Pick", "Year", "Height", "Name", "Unknown1", "Team", "Conference", "Games Playeed", "Role", "MIN%", "PRPG!", "D-PRPG", "BPM", "OBPM", "DBPM", "ORTG", "D-RTG", "USG", "EFG", "TS", "OR", "DR", "AST", "TO", "A/TO", "BLK", "STL", "FTR", "FC/40", "Dunk Ratio", "Dunk %", "Close 2 Ratio", "Close 2 %", "Far 2 Ratio", "Far 2 %", "FT Ratio", "FT %", "2P Ratio", "2P %", "3P/100", "3P Ratio", "3P %"]
    stat_1 = 13
    stat_2 = 21
    stat_1_name = column_list_names[stat_1]
    stat_2_name = column_list_names[stat_2]
    # print(f"player_stats columns names: \n {column_list_names}\n")
    

    name_list = player_stats.iloc[0:num_of_players, 4]
    obpm_list = player_stats.iloc[0:num_of_players, stat_1]
    dbpm_list = player_stats.iloc[0:num_of_players, stat_2]


    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(obpm_list, dbpm_list, alpha=0.5)
    ax.set_title(f'{stat_1_name} vs {stat_2_name} for Players')
    ax.set_xlabel(f'{stat_1_name}')
    ax.set_ylabel(f'{stat_2_name}')

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



def main():

    # TO DO:
    # 1. Import data
    # 2. Figure out how to classify data
    # 2A - We can use clustering to classify players based on their stats. We can use K-Means or Hierarchical Clustering. We can also chart a 2d graph with basic stats. 
    # 3. Save data somewhere

    player_stats = import_data()

    print(f"player_stats info: \n {player_stats[0:5]}\n")

    print(f"player_stats columns: \n {player_stats.columns}\n")

    print(f"player_stats players iloc: \n {player_stats.iloc[0:5, 0:5]}")

    plot_basic_stats(player_stats)

    


    pass

if __name__ == "__main__":
    main()