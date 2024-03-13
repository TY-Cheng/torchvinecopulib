# This is a function to convert the points generated in the tree into strings that meet the requirements.

def process_tree(self):
    # Initialize the list
    lst_tree = [None] * len(self.dct_tree)
    lst_ob = [None] * len(self.dct_tree)
    tree_list = []
    obs_list = []
    arrow_list = []

    for lv in range(len(self.dct_tree)):
        lst_tree[lv] = []
        lst_ob[lv] = []
      
        for v_l, v_r, s_and in self.dct_tree[lv]:
            # Convert an element to a string
            s_and_str = ','.join(str(item) for item in s_and) if s_and else ''
            lst_tree[lv].append(f"{v_l},{v_r}" + (f";{s_and_str}" if s_and_str else ''))
            # get obs
            lst_ob[lv].append(f"{v_l}" + (f"|{s_and_str}" if s_and_str else ''))
            lst_ob[lv].append(f"{v_r}" + (f"|{s_and_str}" if s_and_str else ''))
        # Pack the list of each layer into the master list
        tree_list.append(sorted(list(set(lst_tree[lv]))))  # Use sets to duplicate and sort
        obs_list.append(sorted(list(set(lst_ob[lv]))))  
        
    # get arrow_list
    for lv in range(len(tree_list)):
        for tree in tree_list[lv]:
            tree_set = set([tree[0], tree[2]])  # Extract bit 0 and bit 2 of the string in the tree to form a set
            tree_postfix = tree.split(";")[1] if ";" in tree else ""  
            for obs in obs_list[lv]:
                obs_prefix = obs[0] 
                obs_postfix = obs.split("|")[1] if "|" in obs else ""  
                if obs_prefix in tree_set and obs_postfix == tree_postfix:  
                    arrow_list.append([obs, tree])

        if lv < len(obs_list) - 1:
            for tree in tree_list[lv]:
                tree_set = set(tree.replace(";", ",").split(","))
                for obs in obs_list[lv + 1]:
                    obs_set = set(obs.replace("|", ",").split(","))
                    if tree_set == obs_set:
                        arrow_list.append([tree, obs])
    
    # Duplicate and sort. This operation may not be necessary in fact.
    arrow_list = sorted(list(set(map(tuple, arrow_list))), key=lambda x: (x[0], x[1]))

    # get special_nodes list
    lst_diag = self.diag
    lst_diag = [(v, frozenset(lst_diag[(idx + 1) :])) for idx, v in enumerate(lst_diag)][::-1]
    special_nodes = [f"{v}" + (f"|{','.join(str(item) for item in s)}" if s else '') for v, s in lst_diag]
  
    # return four lists
    return obs_list, tree_list, arrow_list, special_nodes


# The function of drawing according to the input list requirements.
def draw_graph(obs_list, tree_list, arrow_list, special_nodes=None, save_path=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    node_coords = {}  # Dictionary to store the coordinates of each node

    def draw_node(ax, text, center, node_type, node_size=0.4):
        node_coords[text] = center  # Store the coordinates of the node
        edgecolor = 'red' if special_nodes and text in special_nodes else ('blue' if node_type == "obs" else 'green')
        if node_type == "obs":
            circle = plt.Circle(center, node_size, edgecolor=edgecolor, fill=False)
            ax.add_patch(circle)
        else:
            square = plt.Rectangle((center[0] - node_size, center[1] - node_size), node_size*2, node_size*2, edgecolor=edgecolor, fill=False)
            ax.add_patch(square)
        ax.text(center[0], center[1], text, ha='center', va='center', fontsize=12)

    def get_coords(node_name):
        return node_coords.get(node_name, None)  # Return the coordinates of the node, or None if the node does not exist

    def draw_arrow(ax, arrow_list, arrow_color='black', text_color='black', text_alpha=0.8):
        for arrow in arrow_list:
            start, end = arrow[0], arrow[1]
            start_coords = get_coords(start)  # Assume get_coords is a function that returns the coordinates of a node given its name
            end_coords = get_coords(end)
            mid_coords = ((start_coords[0] + end_coords[0]) / 2, (start_coords[1] + end_coords[1]) / 2)  # Calculate the midpoint
            dx = (end_coords[0] - start_coords[0]) / 4  # Half of the distance between the two points
            dy = (end_coords[1] - start_coords[1]) / 4
            arrow_obj = patches.FancyArrow(mid_coords[0] - dx, mid_coords[1] - dy, dx * 2, dy * 2, 
                                       width=0.01, color=arrow_color, shape='full', length_includes_head=True, head_width=0.05)
            ax.add_patch(arrow_obj)
            if len(arrow) > 2:  # If there is a label for the arrow
                label = arrow[2]
                ax.text(mid_coords[0], mid_coords[1], label, ha='center', va='center', fontsize=12, color=text_color, alpha=text_alpha)

    fig, ax = plt.subplots(figsize=(12, 10))  # Adjust the size of the figure
    ax.axis('off')  # Hide the axes

    total_levels = len(obs_list) + len(tree_list)  # The total number of levels

    # Calculate the total width and height of the figure
    total_width = 12
    total_height = 18 #If the canvas is not big enough in actual use, you can modify the parameters here

    for level in range(total_levels):
        obs = obs_list[level] if level < len(obs_list) else []
        tree = tree_list[level] if level < len(tree_list) else []
        # Calculate the interval for each node
        interval_obs = total_width / (len(obs) + 1)
        interval_tree = total_width / (len(tree) + 1)
        # Draw observation nodes
        for i, ob in enumerate(obs):
            draw_node(ax, ob, ((i + 1) * interval_obs, total_height - (level * 2 + 1) * total_height / (total_levels + 1)), 'obs')

        # Draw tree edge nodes
        for i, tr in enumerate(tree):
            draw_node(ax, tr, ((i + 1) * interval_tree, total_height - (level * 2 + 2) * total_height / (total_levels + 1)), 'tree')

    # Adjust the limits of the axes
    ax.set_xlim([0, total_width])
    ax.set_ylim([0, total_height])

    draw_arrow(ax, arrow_list, arrow_color='orange', text_color='black', text_alpha=0.5)

    if save_path is not None:  # If a save path is provided, the image is saved
        plt.savefig(save_path)
    plt.show()  #If you don't need to show the image, you can annotate it.
