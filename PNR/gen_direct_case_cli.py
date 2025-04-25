import cmd
import os
import pandas as pd
import json

from draw_placement import draw_placement

class DirectCaseGenerator(cmd.Cmd):
    prompt = 'canvas>> '
    intro = 'Welcome to DirectCaseGenerator. Type "help" for available commands.'

    def __init__(self):
        super().__init__()
        self.current_directory = os.getcwd()
        self.save_directory = None
        
        self.node_list_df = None
        self.metadata_dict = None
        self.wam_df = None
        self.net_adj_list_df = None
        self.region_df = None

    def do_create_new_canvas(self, arg):
        kwargs = self.parse_args(arg)
        name = kwargs.get('name')
        height = int(kwargs.get('height', 0))
        width = int(kwargs.get('width', 0))

        if not (name and height and width):
            raise ValueError("Incorrect arguments passed {}".format(kwargs))
        # create directory with name
        new_directory = os.path.join(self.current_directory, name)
        os.makedirs(new_directory, exist_ok=True)
        
        # update save_directory with the new directory name
        self.save_directory = new_directory
        print(f"New canvas directory created: {new_directory}")

        # initialize pandas dataframe with column names
        columns = ['node_id', 'node_type', 'width', 'height', 'x', 'y', 'lx', 'ly', 'pin_count', 'cluster_area', 'internal_net', 'group_id', 'region_id', 'place_order']
        self.node_list_df = pd.DataFrame(columns=columns)
        
        # save it as an empty csv under the name {name}_node_list.csv
        node_list_csv_path = os.path.join(new_directory, f"{name}_node_list.csv")
        self.node_list_df.to_csv(node_list_csv_path, index=False)
        
        # create a dictionary with keys canvas_height, canvas_width, cluster_count, macro_count, net_count
        self.metadata_dict = {
            'board info': {
            "canvas_height": height,
            "canvas_width": width,
            "cluster_count": 0,
            "macro_count": 0,
            "net_count": 0,
            "grid_width": 1,
            "grid_height": 1
        }
        }
        
        # save it with name {name}_metadata.json
        metadata_json_path = os.path.join(new_directory, f"{name}_metadata.json")
        with open(metadata_json_path, 'w') as json_file:
            json.dump(self.metadata_dict, json_file)
        
        # create another csv {name}_weighted_adjacency_matrix_sparse.csv which would be empty (no column names) and save it
        self.wam_df = pd.DataFrame()
        wam_csv_path = os.path.join(new_directory, f"{name}_weighted_adjacency_matrix_sparse.csv")
        self.wam_df.to_csv(wam_csv_path, index=False, header=False)
        
        # create another csv {name}_net_adjacency_list.csv which would be empty and save it
        self.net_adj_list_df = pd.DataFrame()
        net_adj_list_csv_path = os.path.join(new_directory, f"{name}_net_adjacency_list.csv")
        self.net_adj_list_df.to_csv(net_adj_list_csv_path, index=False, header = False)
        
        # create a pd.dataframe which contains columns: mask_id, mask_type, lx, ly, ux, uy, region_id and one row: 0, 2, 0, 0, width, height, -1
        region_columns = ['mask_id', 'mask_type', 'lx', 'ly', 'ux', 'uy', 'region_id']
        region_data = [[0, 2, 0, 0, width, height, -1]]
        self.region_df = pd.DataFrame(region_data, columns=region_columns)
        
        # save the dataframe as a csv {name}_region.csv
        region_csv_path = os.path.join(new_directory, f"{name}_region.csv")
        self.region_df.to_csv(region_csv_path, index=False)
        
        draw_placement(self.node_list_df, self.metadata_dict, title='_temp', figure_size=8, font_size=6, connection=self.net_adj_list_df, answer=None,
                   save_file=True, save_file_path=os.path.join(new_directory, "_temp.png"), show=False)

    def do_addNode(self):
        pass
    
    def parse_args(self, arg):
        """Parse a space-separated string of arguments into a dictionary."""
        args = arg.split()
        kwargs = {}
        for arg in args:
            key, value = arg.split('=')
            kwargs[key] = value
        return kwargs

    def do_list(self, line):
        """List files and directories in the current directory."""
        files_and_dirs = os.listdir(self.current_directory)
        for item in files_and_dirs:
            print(item)

    def do_change_dir(self, directory):
        """Change the current directory."""
        new_dir = os.path.join(self.current_directory, directory)
        if os.path.exists(new_dir) and os.path.isdir(new_dir):
            self.current_directory = new_dir
            print(f"Current directory changed to {self.current_directory}")
        else:
            print(f"Directory '{directory}' does not exist.")

    def do_create_file(self, filename):
        """Create a new text file in the current directory."""
        file_path = os.path.join(self.current_directory, filename)
        try:
            with open(file_path, 'w') as new_file:
                print(f"File '{filename}' created in {self.current_directory}")
        except Exception as e:
            print(f"Error: {e}")

    def do_read_file(self, filename):
        """Read the contents of a text file in the current directory."""
        file_path = os.path.join(self.current_directory, filename)
        try:
            with open(file_path, 'r') as existing_file:
                print(existing_file.read())
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
        except Exception as e:
            print(f"Error: {e}")

    def do_quit(self, line):
        """Exit the CLI."""
        return True

    def postcmd(self, stop, line):
        print()  # Add an empty line for better readability
        return stop

if __name__ == '__main__':
 DirectCaseGenerator().cmdloop()