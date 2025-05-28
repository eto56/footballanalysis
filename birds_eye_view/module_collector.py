import ast
import os


visited = set()

summary_map = {}

def extract_imports(filename):

    # get "import" and "from ... import" statements from a python file

    if not os.path.exists(filename):
        #print('No such file')
         return []
    
    if not filename.endswith('.py'):
        filename += '.py'
    #print('Extracting imports from ' + filename)
    with open(filename, 'r') as f:
        tree = ast.parse(f.read(), filename)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    #print(filename + ' imports: ' + str(imports))
    return imports


def get_next_file(filename_list):
    # yolo.model -> yolo/model

    module_list = filename_list.copy()
    path_list = []
    for module in module_list:
        filename_split = module.split('.')
        path = os.path.join(*filename_split)
         
        path_list.append(path)
    
    return path_list

def dfs_search(filename_list):
    # Depth First Search
    # filename_list: list of filenames
    # visited: set of visited filenames
    # graph: dictionary of filename -> list of filenames

    for filename in filename_list:
        if filename in visited:
            continue
        visited.add(filename)

        if not filename.endswith('.py'):
                filename += '.py'
        next_files = get_next_file(extract_imports(filename))
        if next_files==[]:
            continue

        summary_map[filename] = next_files
         
        

        dfs_search(next_files)


def summary():
    for key in summary_map.keys():

        print ('File: ' + key)
        print(str(summary_map[key]))
        print ('-----------------------')

    
    print ('Visited files: ' + str(visited))


if __name__ == '__main__':
    
    tatget_file = "main.py"
     
    dfs_search([tatget_file] )

    summary()

 

 
        

