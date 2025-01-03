import torch
import numpy as np
import scipy.sparse as sp
from torchdiffeq import odeint
import time
# from sparsesvd import sparsesvd

from models.BaseModel import GeneralModel

class BSPM(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['solver_idl','solver_idl','solver_shr','K_idl','T_idl','K_b','T_b','K_s','T_s','factor_dim','idl_beta','final_sharpening','sharpening_off','t_point_combination']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--solver_idl', type=str, default='euler', help="heat equation solver")
        parser.add_argument('--solver_blr', type=str, default='euler', help="ideal low-pass solver")
        parser.add_argument('--solver_shr', type=str, default='euler', help="sharpening solver")

        parser.add_argument('--K_idl', type=int, default=1, help='T_idl / \tau')
        parser.add_argument('--T_idl', type=float, default=1, help='T_idl')

        parser.add_argument('--K_b', type=int, default=1, help='T_b / \tau')
        parser.add_argument('--T_b', type=float, default=1, help='T_b')

        parser.add_argument('--K_s', type=int, default=1, help='T_s / \tau')
        parser.add_argument('--T_s', type=float, default=1, help='T_s')

        parser.add_argument('--factor_dim', type=int, default=256, help='factor_dim')
        parser.add_argument('--idl_beta', type=float, default=0.3, help='beta')

        parser.add_argument('--final_sharpening', type=eval, default=True, choices=[True, False])
        parser.add_argument('--sharpening_off', type=eval, default=False, choices=[True, False])
        parser.add_argument('--t_point_combination', type=eval, default=False, choices=[True, False])

        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.args = args
        self.corpus = corpus
        self.adj_mat = self._construct_adj_mat(self)

        self.idl_solver = args.solver_idl
        self.blur_solver = args.solver_blr
        self.sharpen_solver = args.solver_shr
        print(f"IDL: {self.idl_solver}, BLR: {self.blur_solver}, SHR: {self.sharpen_solver}")

        self.idl_beta = args.idl_beta
        self.factor_dim = args.factor_dim
        print(r"IDL factor_dim: ", self.factor_dim)
        print(r"IDL $\beta$: ", self.idl_beta)
        self.idl_T = args.T_idl
        self.idl_K = args.K_idl

        self.blur_T = args.T_b
        self.blur_K = args.K_b

        self.sharpen_T = args.T_s
        self.sharpen_K = args.K_s

        self.device = torch.device('cuda:'+args.gpu if torch.cuda.is_available() else 'cpu')

        self.idl_times = torch.linspace(0, self.idl_T, self.idl_K + 1).float().to(self.device)
        print("idl time: ", self.idl_times)
        self.blurring_times = torch.linspace(0, self.blur_T, self.blur_K + 1).float().to(self.device)
        print("blur time: ", self.blurring_times)
        self.sharpening_times = torch.linspace(0, self.sharpen_T, self.sharpen_K + 1).float().to(self.device)
        print("sharpen time: ", self.sharpening_times)

        self.final_sharpening = args.final_sharpening
        self.sharpening_off = args.sharpening_off
        self.t_point_combination = args.t_point_combination
        print("final_sharpening: ", self.final_sharpening)
        print("sharpening off: ", self.sharpening_off)
        print("t_point_combination: ", self.t_point_combination)

        self.train_done=False
        self.out_dict = {}

    @staticmethod
    def _construct_adj_mat(self):   #构建用户-物品稀疏矩阵，也就是BSMP源码中的adj_mat
        R = sp.dok_matrix((self.corpus.n_users, self.corpus.n_items), dtype=np.int0)
        train_mat = self.corpus.train_clicked_set
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        adj_mat = R.tolil()

        return adj_mat.tolil()

    def fit(self):    #数据预处理
        adj_mat = self.adj_mat
        print("self.adj_mat:", sum(self.adj_mat.todense()[0, :]))
        # print(adj_mat.todense().shape) # (6033,3126)
        start = time.time()
        rowsum = np.array(adj_mat.sum(axis=1)) + 1e-10
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.

        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)

        colsum = np.array(adj_mat.sum(axis=0)) + 1e-10
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.

        d_mat = sp.diags(d_inv)
        self.d_mat_i = d_mat
        self.d_mat_i_inv = sp.diags(1 / d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.norm_adj = norm_adj.tocsc()
        del norm_adj, d_mat
        ut, s, self.vt = sp.linalg.svds(self.norm_adj, k = self.factor_dim)
        del ut
        del s

        linear_Filter = self.norm_adj.T @ self.norm_adj
        self.linear_Filter = self.convert_sp_mat_to_sp_tensor(linear_Filter).to_dense().to(self.device)

        left_mat = self.d_mat_i @ self.vt.T
        right_mat = self.vt @ self.d_mat_i_inv
        self.left_mat, self.right_mat = torch.FloatTensor(left_mat).to(self.device), torch.FloatTensor(right_mat).to(self.device)
        end = time.time()
        print('pre-processing time for BSPM', end - start)

    def sharpenFunction(self, t, r):
        out = r @ self.linear_Filter
        return -out

    def IDLFunction(self, t, r):
        out = r @ self.left_mat @ self.right_mat
        out = out - r
        return torch.Tensor(out)

    def blurFunction(self, t, r):
        R = self.norm_adj
        out = r @ self.linear_Filter
        out = out - r
        return torch.Tensor(out)

    def predict_for_user(self, feed_dict):
        # 获取用户 ID
        user_ids = feed_dict['user_id'] #type为tensor
        item_ids = feed_dict['item_id']
        batch_test = self.adj_mat[user_ids.cpu(), :]
        batch_test = self.convert_sp_mat_to_sp_tensor(batch_test).to(self.device)

        with torch.no_grad():
            batch_test_dense = batch_test.to_dense()
            idl_out = odeint(func=self.IDLFunction, y0=torch.Tensor(batch_test_dense), t=self.idl_times,
                             method=self.idl_solver)[-1]

            blurred_out = odeint(func=self.blurFunction, y0=torch.Tensor(batch_test_dense), t=self.blurring_times,
                             method=self.blur_solver)[-1]
            del batch_test

            if self.sharpening_off == False:
                if self.final_sharpening == True:
                    sharpened_out = odeint(func=self.sharpenFunction, y0=self.idl_beta * idl_out + blurred_out,
                                            t=self.sharpening_times, method=self.sharpen_solver)
                else:
                    sharpened_out = odeint(func=self.sharpenFunction, y0=blurred_out, t=self.sharpening_times,
                                           method=self.sharpen_solver)
        if self.t_point_combination == True:
            if self.sharpening_off == False:
                U_2 = torch.mean(torch.cat([blurred_out.unsqueeze(0), sharpened_out[1:, ...]], axis=0), axis=0)
            else:
                U_2 = blurred_out
                del blurred_out
        else:
            if self.sharpening_off == False:
                U_2 = sharpened_out[-1]
                del sharpened_out
            else:
                U_2 = blurred_out
                del blurred_out

        if self.final_sharpening == True:
            if self.sharpening_off == False:
                ret = U_2
            elif self.sharpening_off == True:
                ret = self.idl_beta * idl_out + U_2
        else:
            ret = self.idl_beta * idl_out + U_2
        num = ret.shape[0]
        tensor = torch.zeros(num, item_ids.shape[1])
        for i in range(num):
            tensor[i] = (ret[i])[item_ids[i]]
        return tensor


    def forward(self, feed_dict):

        if not self.train_done:
            self.fit()
            self.train_done = True

        self.out_dict['prediction'] = self.predict_for_user(feed_dict)
        return {'prediction': self.out_dict['prediction']}

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return  torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))