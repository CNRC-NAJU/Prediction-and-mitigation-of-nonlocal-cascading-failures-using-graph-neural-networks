# Prediction-and-mitigation-of-nonlocal-cascading-failures-using-graph-neural-networks
Code and data for paper 'Prediction and mitigation of nonlocal cascading failures using graph neural networks'.

In order to fully reproduce the contents of the paper, the following scripts must be executed in order. Alternatively, you can use the simulation results or the trained model we provide right away.

Please refer to the paper for an explanation of each term. Note that in the following code, $\langle C_\sigma (n) \sigma_N$ denoted as MCF (mean cumulative fraction).


# Requirements
- Python codes are executed using python environment in `env.yaml` is required.
- Cpp code is tested at gcc 11.1.0, c++17 standard.


# 1. `construct_shk.py`: Constructing synthetic electric power grid network
Construct synthetic electric power grid following rules introduced at [*A random growth model for power grids and other spatially embedded infrastructure networks*](https://link.springer.com/article/10.1140/epjst/e2014-02279-6) by P. Schultz, J. Heitzig, J. Kurths (2014).

For example, the following run with default arguments will create a `data/edge_list/test_0.txt` file, which is a TSV(tab-separated values) file of the graph edge list.
```
$ python construct_shk.py
```

To find out which arguments are available, run `python construct_shk.py --help` or refer to the following.
```
usage: construct_shk.py [-h] [--initial_num_nodes INITIAL_NUM_NODES] [--num_nodes NUM_NODES] [-p P] [-q Q] [-r R]
                        [-s S] [--ensemble_name ENSEMBLE_NAME] [--ensemble_idx ENSEMBLE_IDX]

options:
  -h, --help            show this help message and exit
  --initial_num_nodes INITIAL_NUM_NODES
                        Number of initial nodes (default: 1)
  --num_nodes NUM_NODES
                        Final number of nodes (default: 100)
  -p P                  Model parameter p (default: 0.2)
  -q Q                  Model parameter q (default: 0.3)
  -r R                  Model parameter r (default: 0.33333)
  -s S                  Model parameter s (default: 0.1)
  --ensemble_name ENSEMBLE_NAME
                        Name of ensemble of networks (default: test)
  --ensemble_idx ENSEMBLE_IDX
                        Ensemble index (default: 0)
```

# 2. `simulate.sh`: Simulating motter-lai model
Run motter-lai model introduced at [*Cascade-based attacks on complex networks*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.66.065102) by A.E. Motter, Y.C. Lai (2002).

This will read the edge list created by `construct_shk.py` and run Motter-Lai model simulation with an initial failure of each node. In the paper, $\alpha=0.25$ is used.
```
$ ./simulate.sh <network_name> <alpha>
```
As the result of the simulation, it will create five TSV files:
- `data/bc/<network_name>`: (N,) vector; Betweenness centrality of each node for a given graph
- `data/bc_one_removed/<network_name>`: (N, N) matrix; Betweenness centrality of each node for a given graph with one node removed
- `data/time/<network_name>_<alpha>.txt`: (N,) vector; Termination time of Motter-Lai model when it is initiated by each node.
- `data/avalanche_fraction/<network_name>_<alpha>.txt`: (N,) vector; Avalanche fraction of each node.
- `data/failure_fraction/<network_name>_<alpha>.txt`: (N,) vector; Failure fraction of each node.

**Note:  The data directories should be created before the code is executed**


# 3. `preprocess.py`: Prepare simulation data for deep learning
This aggregates simulation results into a single `pandas.dataframe` for easy use in training GNN.

The name of the network data to be aggregated and the name of the datafarme created must be modified within the code.

Each row of the dataframe stores the simulation results of single network of $N$ number of nodes, $E$ number of edges.
- edge_index: torch.LongTensor of shape (2, $E$), following `torch_geometric` format
- alpha: torch.Tensor of shape ($N$, ), all constant of value $a$
- avalanche_fraction: torch.Tensor of shape ($N$, )
- failure_fraction: torch.Tensor of shape ($N$, )
- termination_step: torch.LongTensor of shape ($N$, )
- avalanche_centrality: torch.Tensor of shape ($N$, )

The resulting dataframe will be save in pickle format in the `data` directory.
## `data`: Results of motter-lai model
Results of motter-lai model simulation, summarized in `pandas.dataframe` of python.
- Synthetic electric power grids: From `construct_shk.py`
  - `shk_rand_0.25.pkl`: train/validation dataset. SHK network of network size selected at random from $N=100-999$. $10^3$ data are uploaded due to size limit of GitHub(50MB). Larger dataset consisting of $10^4$ data (`shk_rand_10x_0.25.pkl`) is provided upon request.
  - `shk_<network_size>_0.25.pkl`: test dataset. Results of $100$ SHK networks of network size `<network_size>`.
- Real electric power grids: From [*Dynamically induced cascading failures in power grids*](https://www.nature.com/articles/s41467-018-04287-5) by B. Schäfer, D. Witthaut, M. Timme, V. Latora (2018)
  - `es_98_0.25.pkl`: test dataset. Result of Spain power grid network.
  - `fr_146_0.25.pkl`: test dataset. Result of France power grid network.
  - `gb_120_0.25.pkl`: test dataset. Result of Great Britain power grid network.


# 4. `train.py`: Train GNN and test
Train and test GNN with the structure of following figure.
| ![](figure/NN_structure.png)                                                                                                                |
| :------------------------------------------------------------------------------------------------------------------------------------------ |
| Batch normalization and ReLU activation are applied between each adjacent GIN layers, but they are omitted from the diagram for simplicity. |

The GNN and dataset are configured according to the hyperparameters based on user's input. After the training, it will store the training log and results in `experiment/<experiment_id>` directory. The directory has following files
- history.dat: Values of all metrics during training in the order of `train mcf`, `train kendall tau`, `train mse`, `train mae`, `train R2`, `validation mcf`, `validation kendall tau`, `validation mse`, `validation mae`, `validation R2`
- hyper_parameter.json: list of hyperparameters used in training, formatted in json
- debug.log: All log during training and test including loss, times, ...
- real.png: MCF plot of real topological power grids: Spain, France, UK
- scalable.png: MCF plot of synthetic power grids: $N=100-1000, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000$.
- best.pth: Store model states and optimizer states at the best epoch (lowest validation error) during training.
- last.pth: Store model states and optimizer states at the last epoch of training.

To find out which arguments are available, run `python train_gnn.py --help` or refer to the following
```
usage: train.py [-h] [--data_dir DATA_DIR] [--train TRAIN [TRAIN ...]] [--val VAL [VAL ...]]
                    [--add_degree {normalize,log,inverse,sqrt} [{normalize,log,inverse,sqrt} ...]]
                    [--add_network_size {inverse,sqrt,log} [{inverse,sqrt,log} ...]]
                    [--label {avalanche_fraction,failure_fraction,avalanche_centrality}]
                    [--scaler {None,QuantileScaler}] [--model {SkipGINNet}] [--num_layer NUM_LAYER]
                    [--hidden_feature HIDDEN_FEATURE] [--num_parallel NUM_PARALLEL] [--hop_length HOP_LENGTH]
                    [--num_out NUM_OUT] [--out_act {sigmoid,hardsigmoid,softsign,modified_sigmoid}]
                    [--dropout DROPOUT] [--bn_momentum BN_MOMENTUM] [--networkwise] [--loss {MSE,MAE,R2}]
                    [--objective OBJECTIVE] [--optimizer {Adagrad,Adam,RMSprop,SGD}]
                    [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--clip CLIP]
                    [--schedular {Cosine}] [--schedular_period SCHEDULAR_PERIOD]
                    [--schedular_period_mult SCHEDULAR_PERIOD_MULT] [--warmup WARMUP]
                    [--schedular_eta_max SCHEDULAR_ETA_MAX] [--schedular_eta_max_mult SCHEDULAR_ETA_MAX_MULT]
                    [--patience PATIENCE] [--early_stop_delta EARLY_STOP_DELTA]
                    [--device {cpu,cuda:0,cuda:1,cuda:2,cuda:3,0,1,2,3}] [--seed SEED] [--epochs EPOCHS]
                    [--batch_size BATCH_SIZE] [--log_level {debug,info,warning,error,critical}] [--no_tqdm]
                    [--root_dir ROOT_DIR] [--exp_id EXP_ID]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   data path (default: /pds/pds11/hoyun/Motter_Lai_paper/data)
  --train TRAIN [TRAIN ...]
                        Network type for train dataset (default: ['shk_rand_0.25'])
  --val VAL [VAL ...]   Network type for validation dataset (default: ['shk_rand_0.25'])
  --add_degree {normalize,log,inverse,sqrt} [{normalize,log,inverse,sqrt} ...]
                        (deprecated) normalize: [degree/max_degree] log: [log10(degree)] inverse: [1/degree] sqrt:
                        [1/sqrt(degree)] as additional input feature (default: None)
  --add_network_size {inverse,sqrt,log} [{inverse,sqrt,log} ...]
                        (deprecated) inverse: [1/N] sqrt: [1/sqrt(N)] log: [1/log10(N)] as additional input
                        feature (default: None)
  --label {avalanche_fraction,failure_fraction,avalanche_centrality}
                        Which quantity to predict (default: avalanche_centrality)
  --scaler {None,QuantileScaler}
                        Which scaler to use scaling label. If none, predict value, if quantile scaler, predict
                        rank. (default: None)
  --model {SkipGINNet}  GNN model name (default: SkipGINNet)
  --num_layer NUM_LAYER
                        Number of GNN layer (default: 8)
  --hidden_feature HIDDEN_FEATURE
                        Dimension for hidden feature (default: 128)
  --num_parallel NUM_PARALLEL
                        Number of mlp layers used in GIN (default: 2)
  --hop_length HOP_LENGTH
                        How many neighbors to count at once (default: 4)
  --num_out NUM_OUT     Number of layers at regressor (default: 1)
  --out_act {sigmoid,hardsigmoid,softsign,modified_sigmoid}
                        Activation function for out layer (default: sigmoid)
  --dropout DROPOUT     dropout rate (default: 0.0)
  --bn_momentum BN_MOMENTUM
                        Momentum of batch norm (default: 0.1)
  --networkwise         If this flag is not set, metrics are calculated networkwise and then averaged (default:
                        True)
  --loss {MSE,MAE,R2}   Metric name to use as loss: back-propagation (default: MAE)
  --objective OBJECTIVE
                        Metric name for early stopping. Include inv_ in front of metric name if you want to
                        measure after inverse scaling. (default: inv_MCF)
  --optimizer {Adagrad,Adam,RMSprop,SGD}
                        torch optimizer names (default: RMSprop)
  --learning_rate LEARNING_RATE
                        learning rate (default: 0.001)
  --weight_decay WEIGHT_DECAY
                        Weight decay for optimizer (default: 1e-05)
  --clip CLIP           Maximum norm of gradients (default: inf)
  --schedular {Cosine}  Which schedular to control learning rate (default: Cosine)
  --schedular_period SCHEDULAR_PERIOD
                        Period of cosine annealing shedular. When set to 0, schedular will not be used (default:
                        20)
  --schedular_period_mult SCHEDULAR_PERIOD_MULT
                        Multiplier of period of schedular. (default: 2.0)
  --warmup WARMUP       Number of epochs for warm up stage (default: 0)
  --schedular_eta_max SCHEDULAR_ETA_MAX
                        Base eta max used in consine annealing schedular. See detailed description at class
                        CosineAnnealingWarmRestart (default: 0.001)
  --schedular_eta_max_mult SCHEDULAR_ETA_MAX_MULT
                        Multiplier of base eta max. See detailed description at class CosineAnnealingWarmRestart
                        (default: 1.0)
  --patience PATIENCE   How many epochs to wait after validation loss is improved (default: 300)
  --early_stop_delta EARLY_STOP_DELTA
                        Minimum change of validation loss to regard as improved (default: 0.0)
  --device {cpu,cuda:0,cuda:1,cuda:2,cuda:3,0,1,2,3}
                        device to use (default: 0)
  --seed SEED           Seed for torch random (default: None)
  --epochs EPOCHS       Maximum number of epochs (default: 600)
  --batch_size BATCH_SIZE
                        batch size (default: 256)
  --log_level {debug,info,warning,error,critical}
                        Level of logging (default: debug)
  --no_tqdm             When this flag is on, do not use tqdm (default: False)
  --root_dir ROOT_DIR   Root directory for experiments. (default: /pds/pds11/hoyun/Motter_Lai_paper/experiment)
  --exp_id EXP_ID       Experiment index. When specified, load the experiemt (default: None)
```

## 5. `predict.py`: Store prediction of trained GNN
If you want to bring up the GNN in the paper, you can specify `experiment ids` as follows:
```
$ python predict.py --exp_ids 1 2 3 4
```
- exp_ids 1 2 3 4: GNN trained to predict the **rank** of AC of each node in a given network
- exp_ids 5 6 7 8: GNN trained to predict the **value** of AC of each node in a given network
The average of all predictions of GNN will be stored in `data/avalanche_centrality_gnn/<network_name>_<alpha>.txt`.

To find out which arguments are available, run `python predict.py --help` or refer to the following
```
usage: predict.py [-h] [--data_dir DATA_DIR] [--root_dir ROOT_DIR] [--network_name NETWORK_NAME] [--alpha ALPHA]
                  [--exp_ids EXP_IDS [EXP_IDS ...]]

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   data path (default: /pds/pds11/hoyun/Motter_Lai_paper/data)
  --root_dir ROOT_DIR   Root directory for experiments. (default: /pds/pds11/hoyun/Motter_Lai_paper/experiment)
  --network_name NETWORK_NAME
                        Which network to be predicted (default: test_0)
  --alpha ALPHA         alpha used for Motter-Lai model (default: 0.25)
  --exp_ids EXP_IDS [EXP_IDS ...]
                        experiment id of GNNs used for prediction (default: [1, 2, 3, 4])
```

## 6. `mitigate.sh`: Mitigation of cascading failures
Check the effectiveness of avalanche mitigation strategy at motter-lai model.
```
$ ./simulate.sh <network_name> <alpha> <strategy name> <reinforced fraction>
```
Strategy name can be following. Otherwise, it will do nothing
- random
- degree
- bc
- avalanche_fraction
- failure_fraction
- avalanche_centrality
- avalanche_centrality_gnn

As a result, it will store a single number of mean avalanche fraction at `data/mitigation/<network_name>_<alpha>_<strategy name>_<reinforced fraction>.txt`.

**Note: The data directory `data/mitigation` should be created before the code is executed**

## 7. `plot_fig9.ipynb`: Draw figure 9.
An example code to plot figure9-like plot.