from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


class DAGGNN_MLPEncoder(nn.Module):
    """
    MLP encoder module.
    """

    def __init__(self, config, n_feat, in_dim, hidden_dim, out_dim, tol=0.1):
        super().__init__()

        self.n_feat = n_feat
        self.dropout_prob = config.encoder_dropout

        self.adj_A = nn.Parameter(torch.zeros([n_feat, n_feat]))
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

        self.Wa = nn.Parameter(torch.zeros(out_dim))

        # self.z = nn.Parameter(torch.ones(1) * tol)
        # self.z_positive = nn.Parameter(torch.ones([n_feat, n_feat]))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def speed_up(self):
        return torch.sinh(3. * self.adj_A)

    def forward(self, inputs):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = self.speed_up()

        # adj_A for z = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1)  # [F, F]

        H1 = F.relu((self.fc1(inputs)))  # [B, F, Di] -> [B, F, Dh]
        x = self.fc2(H1)  # [B, F, Dh] -> [B, F, Do]
        # [F, F] @ [B, F, Do] = [B, F, Do] through boardcasting
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        # return x, logits, adj_A1, self.z, self.z_positive, self.adj_A, self.Wa
        return x, logits, adj_A1, self.adj_A, self.Wa


class DAGGNN_MLPDecoder(nn.Module):
    """
    MLP decoder module.
    """

    def __init__(self, config, n_feat, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.n_feat = n_feat
        self.dropout_prob = config.decoder_dropout

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_z, origin_A, Wa):

        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa

        H3 = F.relu(self.fc1((mat_z)))
        out = self.fc2(H3)

        return mat_z, out


class DAGGNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.recursive_dag_search = config.recursive_dag_search
        self.n_feat = config.num_feature_nodes

        self.c_A = config.c_A
        self.lambda_A = config.lambda_A
        self.tau_A = config.tau_A
        self.graph_threshold = config.graph_threshold

        self.x_dim = config.x_dim
        self.encoder_hidden_dim = config.encoder_hidden_size
        self.decoder_hidden_dim = config.decoder_hidden_size
        self.z_dim = config.z_dim

        self.elbo_lambda = config.elbo_lambda
        self.rec_lambda = config.rec_lambda

        # Load Forbidden Edges
        self.forbidden_edges = load_forbidden_edges(config)
        # Load Required Edges
        self.positive_required = load_required_edges(config, positive=True)
        self.negative_required = load_required_edges(config, positive=False)

        self.encoder = DAGGNN_MLPEncoder(
            config=config,
            n_feat=self.n_feat,
            in_dim=self.x_dim,
            hidden_dim=self.encoder_hidden_dim,
            out_dim=self.z_dim
        )

        self.decoder = DAGGNN_MLPDecoder(
            config=config,
            n_feat=self.n_feat,
            in_dim=self.z_dim,
            hidden_dim=self.decoder_hidden_dim,
            out_dim=self.x_dim
        )

        self.prox_plus = nn.Threshold(0., 0.)
        self.graph = None

    def _h_A(self, A, m):
        expm_A = matrix_poly(A * A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    def stau(self, w, tau):
        w1 = self.prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1

    def forward(self, x: torch.Tensor):
        # enc_x, logits, origin_A, z_gap, z_positive, myA, Wa = self.encoder(x)
        enc_x, logits, origin_A, myA, Wa = self.encoder(x)
        edges = logits
        dec_x, output = self.decoder(edges, origin_A, Wa)

        self.origin_A = origin_A
        self.myA = myA

        if torch.sum(output != output):
            print('nan error \n')

        targets = x
        preds = output

        variance = 0.
        loss = self.loss_fn(
            preds=preds, targets=targets, variance=variance, logits=logits, graph=origin_A, 
            # z_gap=z_gap, z_positive=z_positive
        )

        # # ==============================================
        # # Enforce all the Prior Knowledge
        # # ==============================================
        # self.origin_A = enforce_edge_constraints(
        #     self.origin_A, self.forbidden_edges, self.positive_required, self.negative_required, self.graph_threshold
        # )
        # self.myA.data = self.stau(self.myA.data, self.tau_A * 5e-5)

        # if torch.sum(self.origin_A != self.origin_A):
        #     print("nan error\n")

        # self.graph = self.origin_A.data.clone()
        # # self.graph.diagonal().zero_()
        # self.graph = self.graph.cpu().numpy()

        return loss

    def graph_refinement(self):
        #! 想办法改在optim.step之后进行
        # ==============================================
        # Enforce all the Prior Knowledge
        # ==============================================
        self.origin_A = enforce_edge_constraints(
            self.origin_A, self.forbidden_edges, self.positive_required, self.negative_required, self.graph_threshold
        )
        self.myA.data = self.stau(self.myA.data, self.tau_A * 5e-5)

        if torch.sum(self.origin_A != self.origin_A):
            print("nan error\n")

        self.graph = self.origin_A.data.clone()
        # self.graph.diagonal().zero_()
        self.graph = self.graph.cpu().numpy()

    # def loss_fn(self, preds, targets, variance, logits, graph, z_gap, z_positive):
    def loss_fn(self, preds, targets, variance, logits, graph):
        # reconstruction accuracy loss, which uses negative log-likelihood of Gaussian prior
        loss_nll = nll_gaussian(preds, targets, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss = KL loss + NLL loss
        loss_elbo = loss_kl + loss_nll

        # Sparsity loss, which uses L1 norm of A
        one_adj_A = graph  # torch.mean(adj_A_tilt_decoder, dim=0)
        #! adjust tau_A to find a proper graph
        loss_sparse = self.tau_A * torch.sum(torch.abs(one_adj_A))

        h_A = self._h_A(graph, self.n_feat)
        # loss_lagr = f(A, theta) + lambda * h(A) + 0.5 * c * h(A)^2
        #! add large c_A and lambda_A, example 100, 100
        #! 关注lagr和elbo的量级
        #! 效果不好的话采用prompt forbidden
        loss_lagr = 100. * torch.trace(graph * graph) + self.lambda_A * h_A + 0.5 * self.c_A * h_A * h_A

        #! add lambda reconstruction
        loss_mse = F.mse_loss(preds, targets)

        loss = self.elbo_lambda * loss_elbo + loss_sparse + loss_lagr + self.rec_lambda * loss_mse
        output = {
            "loss_nll": loss_nll.item(),
            "loss_kl": loss_kl.item(),
            "loss_elbo": loss_elbo.item(),
            "loss_sparse": loss_sparse.item(),
            "loss_lagr": loss_lagr.item(),
            "loss_mse": loss_mse.item(),
        }

        # if self.config.use_A_connect_loss:
        #     connect_gap = A_connect_loss(one_adj_A, self.graph_threshold, z_gap)
        #     loss_connect = self.lambda_A * connect_gap + 0.5 * self.c_A * connect_gap * connect_gap
        #     loss += loss_connect
        #     output["loss_connect"] = loss_connect.item()
        # else:
        #     output["loss_connect"] = None

        # if self.config.use_A_positiver_loss:
        #     positive_gap = A_positive_loss(one_adj_A, self.positive_required, self.negative_required, z_positive)
        #     loss_positive = 0.1 * (self.lambda_A * positive_gap + 0.5 * self.c_A * positive_gap * positive_gap)
        #     loss += loss_positive
        #     output["loss_positive"] = loss_positive.item()
        # else:
        #     output["loss_positive"] = None

        output["loss"] = loss
        output["loss_graph"] = loss.item()
        return output

    def _get_dag(self, As: np.ndarray) -> np.ndarray:
        As_temp = np.abs(As.copy())

        As_temp[np.where(As_temp == As_temp[As_temp > 0].min())] = 0

        intermediate_dag = As_temp.copy()
        intermediate_dag[intermediate_dag > 0] = 1

        if is_dag(intermediate_dag):
            return intermediate_dag
        else:
            return self._get_dag(As_temp)

    def get_A(self, threshold) -> np.ndarray:
        B_est = self.graph.copy()

        if self.recursive_dag_search:
            B_est = self._get_dag(B_est)
        else:
            B_est[np.abs(B_est) <= threshold] = 0
            #! 应该等于graph的值
            # B_est[np.abs(B_est) > threshold] = 1

        return B_est

    def auto_get_dag(self) -> Any:
        if self.graph is None:
            origin_A = self.encoder.speed_up()
            origin_A = enforce_edge_constraints(
                origin_A, self.forbidden_edges, self.positive_required, self.negative_required, self.graph_threshold
            )
            self.graph = origin_A.data.clone()
            self.graph = self.graph.cpu().numpy()

        graphs = {}
        #! 把所有满足的都记录下来
        for threshold in np.linspace(start=0, stop=1, num=100):
            B_est = self.get_A(threshold)
            if is_dag(B_est):
                print(f"Is DAG for {threshold}")
                graphs[threshold] = B_est

        if len(graphs) == 0:
            print("No DAG found")

        graphs["raw"] = self.graph
        return graphs

