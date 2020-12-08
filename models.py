import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import wandb
from tqdm import tqdm

# ------------------------------------------------------------
class Recommender(nn.Module):
    def __init__(self, model, name):
        super().__init__()
        self.model = model
        self.model_name = name
        if self.model_name != 'BPR':
            self.u = self.model.u
            self.v = self.model.v

    def fit(self, data, params, plot=False):
        if self.model_name == 'BPR':
            epochs = params['epochs']
            self.model.fit_partial(data, epochs=epochs)
            self.u = self.model.user_embeddings
            self.v = self.model.item_embeddings
        else:
            epochs, lr, batch_size = params['epochs'], params['lr'], params['batch_size']
            if plot:
                wandb.init(project='ncf', name=self.model_name, config=params)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr)
            trainloader = torch.utils.data.DataLoader(data,
                                                      batch_size=batch_size,
                                                      shuffle=True)
            for epoch in tqdm(range(epochs)):
                losses = []
                self.model.train()
                for x in trainloader:
                    user = x[:, 0].to(device)
                    item = x[:, 1].to(device)
                    label = x[:, 2].float().to(device)

                    self.model.zero_grad()
                    prediction = self.model(user, item)
                    loss = criterion(prediction, label)
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                if plot:
                    wandb.log({"loss": np.mean(losses)})
                # print(f'Epoch {epoch} - Loss: {np.mean(losses)}')

            self.u = self.model.u
            self.v = self.model.v

    def predict(self, user, item):
        return self.model.predict(user, item)

    def recommend(self, user, negative, k=10):
        neg = negative[user]
        if self.model_name == 'BPR':
            scores = self.v[neg] @ self.u[user]
        else:
            scores = self.model.predict(user * np.ones(len(neg)), neg)
            scores = scores.cpu().detach().numpy()
        ids = np.argsort(scores)[::-1][:k]
        return np.array(neg)[ids]

    def similar_items(self, item, k=10):
        norm = np.linalg.norm(self.v, axis=-1)
        norm[norm == 0] = 1e-10
        scores = self.v @ self.v[item] / norm
        ids = np.argsort(scores)[::-1][:k]
        return ids

# ----------------------- NCF ------------------------------
class GFM(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.users_embeddings = nn.Embedding(n_users, dim)
        self.items_embeddings = nn.Embedding(n_items, dim)
        self.f = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

        self.u = self.users_embeddings.weight.detach().cpu().numpy()
        self.v = self.items_embeddings.weight.detach().cpu().numpy()

    def forward(self, user, item):
        user = self.users_embeddings(user)
        item = self.items_embeddings(item)
        out = user * item
        out = self.f(out).view(-1)
        return out

    def predict(self, user, item):
        user = torch.tensor(user).to(device).long()
        item = torch.tensor(item).to(device).long()
        return self(user, item)


class MLP(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, hidden_dim):
        super().__init__()
        self.users_embeddings = nn.Embedding(n_users, embed_dim)
        self.items_embeddings = nn.Embedding(n_items, embed_dim)

        layers = [nn.Linear(embed_dim * 2, hidden_dim[0])]
        for dim_in, dim_out in zip(hidden_dim[:-1], hidden_dim[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        self.f = nn.Sequential(nn.Linear(hidden_dim[-1], 1), nn.Sigmoid())

        self.u = self.users_embeddings.weight.detach().cpu().numpy()
        self.v = self.items_embeddings.weight.detach().cpu().numpy()

    def forward(self, user, item):
        user = self.users_embeddings(user)
        item = self.items_embeddings(item)
        out = torch.cat((user, item), -1)
        out = self.layers(out)
        out = self.f(out).view(-1)
        return out

    def predict(self, user, item):
        user = torch.tensor(user).to(device).long()
        item = torch.tensor(item).to(device).long()
        return self(user, item)


class NeuCF(nn.Module):
    def __init__(self, n_users, n_items, dim_gfm, dim_mlp, hidden_dim, gfm_path, mlp_path):
        super().__init__()
        self.gfm = GFM(n_users, n_items, dim_gfm).to(device)
        self.mlp = MLP(n_users, n_items, dim_mlp, hidden_dim).to(device)
        self.f = nn.Sequential(nn.Linear(hidden_dim[-1] + dim_gfm, 1), nn.Sigmoid())

        self.gfm.load_state_dict(torch.load(gfm_path, map_location=torch.device('cpu')))
        self.mlp.load_state_dict(torch.load(mlp_path, map_location=torch.device('cpu')))

        self.u = np.hstack((self.gfm.users_embeddings.weight.detach().cpu().numpy(),
                            self.mlp.users_embeddings.weight.detach().cpu().numpy()))
        self.v = np.hstack((self.gfm.items_embeddings.weight.detach().cpu().numpy(),
                            self.mlp.items_embeddings.weight.detach().cpu().numpy()))

    def forward(self, user, item):
        user_gfm = self.gfm.users_embeddings(user)
        item_gfm = self.gfm.items_embeddings(item)
        out_gfm = user_gfm * item_gfm

        user_mlp = self.mlp.users_embeddings(user)
        item_mlp = self.mlp.items_embeddings(item)
        out_mlp = torch.cat((user_mlp, item_mlp), -1)
        out_mlp = self.mlp.layers(out_mlp)

        out = torch.cat((out_gfm, out_mlp), -1)
        out = self.f(out).view(-1)
        return out

    def predict(self, user, item):
        user = torch.tensor(user).to(device).long()
        item = torch.tensor(item).to(device).long()
        return self(user, item)
