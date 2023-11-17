import os

# persistent_folder = os.getenv('PERSISTENT_FOLDER', None)
workspace = os.getenv('WORKSPACE', None)

c.ServerApp.terminado_settings = {'shell_command': ['/bin/zsh']}

# c.ServerProxy.servers = {
#     "code-server": {
#         "command": [
#             "code-server",
#             "--auth=none",
#             "--disable-telemetry",
#             "--host=127.0.0.1",
#             "--port={port}",
#             os.getenv("JUPYTER_SERVER_ROOT", ".")
#         ],
#         "timeout": 20,
#         "launcher_entry": {
#             "title": "VisualStudio Code",
#             "icon_path": "/etc/jupyter/vscode.svg",
#             "enabled": True
#         },
#     },
# }

# if persistent_folder:
#     os.chdir(persistent_folder)


# if os.path.exists('packages.txt'):
#     os.system('sudo apt-get update')
#     os.system('cat packages.txt | xargs sudo apt-get install -y')

if os.path.exists('requirements.txt'):
    os.system('pip install -r requirements.txt')

# if os.path.exists('extensions.txt'):
#     os.system('cat extensions.txt | xargs -I {} jupyter {} install --user')

# # Conda install getting stuck sometimes
# if os.path.exists('environment.yml'):
#     os.system('mamba env create -f environment.yml')
#     # os.system('conda env create -f environment.yml')

# if os.path.exists('environment.yaml'):
#     os.system('mamba env create -f environment.yaml')
#     # os.system('conda env create -f environment.yaml')

if workspace:
    os.chdir(workspace)

