## PyCharm example:

# What the team uses:
runai submit \
  --interactive \
  --name no-representation-no-trust-dev \
  --image registry.rcp.anonymous.xyz/anonymous/anonymous/no-representation-no-trust:dev-latest-anonymous \
  --pvc runai-anonymous-anonymous-scratch:/anonymous-rcp-scratch \
  -e PROJECT_ROOT_AT=/anonymous-rcp-scratch/home/anonymous/no-representation-no-trust/dev \
  -e SSH_SERVER=1 \
  -e PYCHARM_IDE_AT=/anonymous-rcp-scratch/home/anonymous/remote-development/pycharm \
  -e PYCHARM_CONFIG_AT=/anonymous-rcp-scratch/home/anonymous/remote-development/pycharm-config \
  -e WANDB_API_KEY_FILE_AT=/anonymous-rcp-scratch/home/anonymous/.wandb-api-key \
  -e GIT_CONFIG_AT=/anonymous-rcp-scratch/home/anonymous/remote-development/gitconfig \
  -e OMP_NUM_THREADS=1 \
  -g 1 --cpu 10 --large-shm \
  -- sleep infinity

# Option 1. No PyCharm on the remote server. Launch PyCharm from your local machine.
runai submit \
  --name example-remote-development \
  --interactive \
  --image registry.rcp.anonymous.xyz/anonymous/anonymous/no-representation-no-trust:dev-latest-anonymous \
  --pvc runai-anonymous-anonymous-scratch:/anonymous-rcp-scratch \
  -e PROJECT_ROOT_AT=/anonymous-rcp-scratch/home/anonymous/no-representation-no-trust/dev \
  -e SSH_SERVER=1 \
  -e PYCHARM_CONFIG_AT=/anonymous-rcp-scratch/home/anonymous/remote-development/pycharm-config \
  -- sleep infinity

# Option 2 (preferred). PyCharm launched from the remote server.
runai submit \
  --name example-remote-development \
  --interactive \
  --image registry.rcp.anonymous.xyz/anonymous/anonymous/no-representation-no-trust:dev-latest-anonymous \
  --pvc runai-anonymous-anonymous-scratch:/anonymous-rcp-scratch \
  -e PROJECT_ROOT_AT=/anonymous-rcp-scratch/home/anonymous/no-representation-no-trust/dev \
  -e SSH_SERVER=1 \
  -e PYCHARM_IDE_AT=/anonymous-rcp-scratch/home/anonymous/remote-development/pycharm \
  -e PYCHARM_CONFIG_AT=/anonymous-rcp-scratch/home/anonymous/remote-development/pycharm-config \
  -- sleep infinity

## The new bits here are:
# -e JETBRAINS_CONFIG_AT=<> will be mapped to ~/.config/JetBrains in the container
# -e PYCHARM_IDE_AT=<> starts the IDE from the container directly.

## VS Code example:
runai submit \
  --name example-remote-development \
  --interactive \
  --image registry.rcp.anonymous.xyz/anonymous/anonymous/no-representation-no-trust:dev-latest-anonymous \
  --pvc runai-anonymous-anonymous-scratch:/anonymous-rcp-scratch \
  -e PROJECT_ROOT_AT=/anonymous-rcp-scratch/home/anonymous/no-representation-no-trust/dev \
  -e SSH_SERVER=1 \
  -e VSCODE_CONFIG_AT=/anonymous-rcp-scratch/home/anonymous/remote-development/vscode-server \
  -- sleep infinity

## The new bits here are:
# -e VSCODE_CONFIG_AT=<> will be mapped to ~/.vscode-server in the container

## Jupyter Lab example:
runai submit \
  --name example-remote-development \
  --interactive \
  --image registry.rcp.anonymous.xyz/anonymous/anonymous/no-representation-no-trust:dev-latest-anonymous \
  --pvc runai-anonymous-anonymous-scratch:/anonymous-rcp-scratch \
  -e PROJECT_ROOT_AT=/anonymous-rcp-scratch/home/anonymous/no-representation-no-trust/dev \
  -e JUPYTER_SERVER=1 \
  -- sleep infinity

## The new bits here are:
# -e JUPYTER_SERVER=1 will start a Jupyter Lab server in the container.

## Useful commands.
# runai describe job example-remote-development
# runai logs example-remote-development
# kubectl port-forward example-remote-development-0-0  2222:22
# ssh runai
# kubectl port-forward example-remote-development-0-0  8888:8888
# runai logs example-remote-development
# Get the link and paste it in your browser, replacing hostname with localhost.

## Troubleshooting.
# When you add a new line for an environment variable or a GPU, etc., remember to add a \ at the end of the line.
# ... \
# -e SOME_ENV_VAR=1 \
# -g 1 \
#...
# -- sleep infinity
