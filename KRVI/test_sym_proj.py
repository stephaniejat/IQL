import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.linalg import block_diag
from torch.utils.data import Dataset, DataLoader
import mlflow
import copy


from KRVI.sym_proj import LinearProjLayer
from KRVI.group_rep import GroupRep


def construct_90deg_block_rot_groups(dim_space: int):
    assert dim_space % 2 == 0
    rotation_matrix = np.array([[0, -1], 
                            [1, 0]])

    blocks = [rotation_matrix]*int(dim_space / 2)
    block_diagonal_matrix = block_diag(*blocks)

    rot_group_2d = [np.eye(2), rotation_matrix, rotation_matrix @ rotation_matrix,  rotation_matrix @ rotation_matrix @ rotation_matrix] 
    rot_group_nd = [np.eye(dim_space), block_diagonal_matrix, block_diagonal_matrix @ block_diagonal_matrix, block_diagonal_matrix @ block_diagonal_matrix @ block_diagonal_matrix]

    return GroupRep(rot_group_nd), GroupRep(rot_group_2d)

def assert_equal_torch(t1, t2):
    assert np.allclose(t1.detach().numpy(), t2.detach().numpy())

class RandomDataset(Dataset):
    def __init__(self, num_samples, num_features, group):
        self.num_samples = num_samples
        self.num_features = num_features
        self.data = torch.randn(num_samples, num_features).to(torch.float)
        self.target = torch.tensor([target_function(sample) for sample in self.data], dtype=torch.float)
        self.invariant_target = torch.tensor([invariant_target(sample, group) for sample in self.data], dtype=torch.float)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], self.invariant_target[idx]

def target_function(x):
    # As proposed by ChatGPT
    result = (2 * x[0] + 3 * x[1]**2 - x[2] * x[3] + torch.sin(x[4]) +
              torch.log1p(np.abs(x[5])) - x[6]**3 + torch.exp(x[7]) + x[8] * x[9] -
              x[10] / (np.abs(x[11]) + 1) + torch.sum(x))
    
    return result

def invariant_target(x, group):
    out = 0
    for g in group.transformations:
        out += target_function(g @ x)
    return out / len(group.transformations)

def train_eval_feedforward_model(model, learning_rate, num_epochs, train_loader, test_loader, run_name, tr_type="autoencoder"):
    # reconstruction_loss 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Start an MLflow run
    with mlflow.start_run(run_name = run_name):
        # Log parameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets, inv_targets in train_loader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward passe
                outputs = model(inputs.cuda())
                if tr_type =="autoencoder":
                    targets = inputs.clone().detach()
                elif tr_type == "inv_regression":
                    targets == inv_targets
                    outputs = outputs.flatten()
                elif tr_type == "regression":
                    outputs = outputs.flatten()
                loss = criterion(outputs.cuda(), targets.cuda())

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)
                

            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for inputs, targets, inv_targets in test_loader:

                    outputs = model(inputs.cuda())
                    if tr_type =="autoencoder":
                        targets = inputs.clone().detach()
                    elif tr_type == "inv_regression":
                        targets == inv_targets
                        outputs = outputs.flatten()
                    elif tr_type == "regression":
                        outputs = outputs.flatten()
                        pass
                    loss = criterion(outputs.cuda(), targets.cuda())
                    total_loss += loss.item()

                avg_loss = total_loss / len(test_loader)
                mlflow.log_metric("test_loss", avg_loss, step=epoch)
                print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    DIM_IN = 12
    DIM_OUT = 2
    G12, G2 = construct_90deg_block_rot_groups(DIM_IN)
    G10, G2 = construct_90deg_block_rot_groups(10)
    G8, G2 = construct_90deg_block_rot_groups(8)
    G6, G2 = construct_90deg_block_rot_groups(6)
    G4, G2 = construct_90deg_block_rot_groups(4)

    # Verify Layer is equivariant
    gell = LinearProjLayer(DIM_IN, DIM_OUT, G12, G2, bias = True)
    x = torch.randn(DIM_IN, dtype = torch.float)
    for index in range(4):
        assert_equal_torch(G2[index] @ gell(x), gell(G12[index] @ x))

    # Set up data
    train_samples = 4000
    test_samples = 500
    num_features = DIM_IN
    batch_size = 128
    trainset = RandomDataset(train_samples, num_features, G12)
    testset = RandomDataset(test_samples, num_features, G12)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    # Non-equivariant baseline
    linear_autoencoder = nn.Sequential(
            nn.Linear(DIM_IN, 2, bias = True),
            nn.Linear(2, DIM_IN, bias = True),
        ).to(torch.float)

    # Just a linear mapping with a bottleneck, can view it as equivariant PCA
    eqv_linear_autoencoder = nn.Sequential(
            LinearProjLayer(DIM_IN, 2, G12, G2, bias = True),
            LinearProjLayer(2, DIM_IN, G2, G12, bias = True),
        ).to(torch.float)
    
    # ReLU activated equivariant autoencoder
    non_linear_autoencoder = nn.Sequential(
        LinearProjLayer(12, 10, G12, G10, bias = True),
        nn.ReLU(),
        LinearProjLayer(10, 8, G10, G8, bias = True),
        nn.ReLU(),
        LinearProjLayer(8, 6, G8, G6, bias = True),
        nn.ReLU(),
        LinearProjLayer(6, 4, G6, G4, bias = True),
        nn.ReLU(),
        LinearProjLayer(4, 2, G4, G2, bias = True),
        nn.ReLU(),
        LinearProjLayer(2, 4, G2, G4, bias = True),
        nn.ReLU(),
        LinearProjLayer(4, 6, G4, G6, bias = True),
        nn.ReLU(),
        LinearProjLayer(6, 8, G6, G8, bias = True),
        nn.ReLU(),
        LinearProjLayer(8, 10, G8, G10, bias = True),
        nn.ReLU(),
        LinearProjLayer(10, 12, G10, G12, bias = True),
        nn.ReLU(),
    )

    regressor_base1 = nn.Sequential(
        nn.Linear(12, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).cuda()
    regressor_base2 = copy.deepcopy(regressor_base1) #TODO: implement correctly

    # Regressor with equivariant linear projection layer
    regressor1 = nn.Sequential(
        # LinearProjLayer(DIM_IN, DIM_OUT, G12, G2, bias = True),
        # nn.ReLU(),
        nn.Linear(12, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    ).cuda()
    regressor2 = copy.deepcopy(regressor1) #TODO: implement correctly
    # Regressor with non-linear equivariant backbone
    
    regressor3 = nn.Sequential(
        LinearProjLayer(12, 10, G12, G10, bias = True),
        nn.ReLU(),
        LinearProjLayer(10, 8, G10, G8, bias = True),
        nn.ReLU(),
        LinearProjLayer(8, 6, G8, G6, bias = True),
        nn.ReLU(),
        LinearProjLayer(6, 4, G6, G4, bias = True),
        nn.ReLU(),
        LinearProjLayer(4, 2, G4, G2, bias = True),
        nn.ReLU(),
        nn.Linear(2, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
    regressor4 = copy.deepcopy(regressor3) #TODO: implement correctly

    learning_rate = 0.001
    num_epochs = 20

    # train_eval_feedforward_model(linear_autoencoder, learning_rate, num_epochs, train_loader, test_loader, run_name = "linear_ae")
    # train_eval_feedforward_model(eqv_linear_autoencoder, learning_rate, num_epochs, train_loader, test_loader, run_name = "eqv_lin_ae")
    # train_eval_feedforward_model(non_linear_autoencoder, learning_rate, num_epochs, train_loader, test_loader, run_name = "eqv_nonlin_ae")
    train_eval_feedforward_model(regressor_base1, learning_rate, num_epochs, train_loader, test_loader, run_name = "reg_base", tr_type = "regression")
    # train_eval_feedforward_model(regressor_base2, learning_rate, num_epochs, train_loader, test_loader, run_name = "inv_reg_base", tr_type = "inv_regression")
    train_eval_feedforward_model(regressor1, learning_rate, num_epochs, train_loader, test_loader, run_name = "reg_shallow", tr_type = "regression")
    # train_eval_feedforward_model(regressor2, learning_rate, num_epochs, train_loader, test_loader, run_name = "inv_reg_shallow", tr_type = "inv_regression")
    # train_eval_feedforward_model(regressor3, learning_rate, num_epochs, train_loader, test_loader, run_name = "reg_deep", tr_type = "regression")
    # train_eval_feedforward_model(regressor4, learning_rate, num_epochs, train_loader, test_loader, run_name = "inv_reg_deep", tr_type = "inv_regression")
    

if __name__ == "__main__":
    main()