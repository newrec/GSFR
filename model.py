from torch import nn
import torch.nn.functional as F
import torch
torch.backends.cudnn.enabled = False

class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()

        self.mlp = nn.Sequential(

            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.aggre(x)


class _UserModel(nn.Module):


    def __init__(self, emb_dim, user_emb, item_emb, rate_emb):
        super(_UserModel, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.emb_dim = emb_dim


        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w4 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w5 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w6 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w7 = nn.Linear(self.emb_dim, self.emb_dim)


        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_items_att_s1 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items_s1 = _Aggregation(self.emb_dim, self.emb_dim)


        self.u_user_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.u_aggre_neigbors = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_users_att_s2 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_neigbors_s2 = _Aggregation(self.emb_dim, self.emb_dim)


        self.combine_mlp = nn.Sequential(
            nn.Linear(3 * self.emb_dim, 2 * self.emb_dim, bias=True),
            nn.BatchNorm1d(2 * self.emb_dim, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.BatchNorm1d(self.emb_dim, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eps = 1e-10

    def forward(self, uids, u_item_pad, u_user_pad, u_user_item_pad):


        q_a = self.item_emb(u_item_pad[:, :, 0])

        mask_u = torch.where(u_item_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))

        u_item_er = self.rate_emb(u_item_pad[:, :, 1])

        x_ia = self.g_v(torch.cat([q_a, u_item_er], dim=2).view(-1, 2 * self.emb_dim)).view(
            q_a.size())


        p_i = mask_u.unsqueeze(2).expand_as(x_ia) * self.user_emb(uids).unsqueeze(1).expand_as(
            q_a)  #
        alpha = self.user_items_att(torch.cat([self.w1(x_ia), self.w1(p_i)], dim=2).view(-1, 2 * self.emb_dim)).view(
            mask_u.size())
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)


        h_iI = self.aggre_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ia) * x_ia, 1))  # B x emb_dim
        h_iI = F.dropout(h_iI, p=0.5, training=self.training)





        q_a_s = self.item_emb(u_user_item_pad[:, :, :, 0])
        mask_s = torch.where(u_user_item_pad[:, :, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))  #

        u_user_item_er = self.rate_emb(u_user_item_pad[:, :, :, 1])  # B x maxu_len x maxi_len x emb_dim
        x_ia_s = self.g_v(torch.cat([q_a_s, u_user_item_er], dim=3).view(-1, 2 * self.emb_dim)).view(
            q_a_s.size())
        p_i_s = mask_s.unsqueeze(3).expand_as(x_ia_s) * self.user_emb(u_user_pad).unsqueeze(2).expand_as(
            q_a_s)

        alpha_s = self.user_items_att_s1(
            torch.cat([self.w4(x_ia_s), self.w4(p_i_s)], dim=3).view(-1, 2 * self.emb_dim)).view(
            mask_s.size())
        alpha_s = torch.exp(alpha_s) * mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)


        h_oI_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ia_s) * x_ia_s, 2)
        h_oI = self.aggre_items_s1(h_oI_temp.view(-1, self.emb_dim)).view(h_oI_temp.size())
        h_oI = F.dropout(h_oI, p=0.5, training=self.training)


        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device),
                              torch.tensor([0.], device=self.device))

        beta = self.user_users_att_s2(
            torch.cat([self.w5(h_oI), self.w5(p_i)], dim=2).view(-1, 2 * self.emb_dim)).view(
            mask_su.size())
        beta = torch.exp(beta) * mask_su
        beta = beta / (torch.sum(beta, 1).unsqueeze(1).expand_as(beta) + self.eps)
        h_iS = self.aggre_neigbors_s2(torch.sum(beta.unsqueeze(2).expand_as(h_oI) * h_oI, 1))
        h_iS = F.dropout(h_iS, p=0.5, training=self.training)




        h_dot = torch.mul(h_iI,h_iS)
        h_cat = torch.cat([h_iI,h_iS],dim=1)
        h = self.combine_mlp(torch.cat([h_dot,h_cat], dim=1))

        return h


class _ItemModel(nn.Module):



    def __init__(self, emb_dim, user_emb, item_emb, rate_emb):
        super(_ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb

        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w4 = nn.Linear(self.emb_dim, self.emb_dim)

        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)

        self.item_users_att_i = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users_i = _Aggregation(self.emb_dim, self.emb_dim)

        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)


        self.item_users_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users = _Aggregation(self.emb_dim, self.emb_dim)

        self.i_friends_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_i_friends = _Aggregation(self.emb_dim, self.emb_dim)

        self.if_friends_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_if_friends = _Aggregation(self.emb_dim, self.emb_dim)

        self.combine_mlp = nn.Sequential(
            nn.Linear(3 * self.emb_dim, 2 * self.emb_dim, bias=True),
            nn.BatchNorm1d(2 * self.emb_dim, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.BatchNorm1d(self.emb_dim, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.emb_dim, self.emb_dim, bias=True)
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.eps = 1e-10

    def forward(self, iids, i_user_pad,uids, u_item_pad):

        p_t = self.user_emb(i_user_pad[:, :, 0])

        mask_i = torch.where(i_user_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        i_user_er = self.rate_emb(i_user_pad[:, :, 1])
        f_jt = self.g_u(torch.cat([p_t, i_user_er], dim=2).view(-1, 2 * self.emb_dim)).view(p_t.size())


        q_j = mask_i.unsqueeze(2).expand_as(f_jt) * self.item_emb(iids).unsqueeze(1).expand_as(f_jt)

        miu = self.item_users_att_i(torch.cat([self.w1(p_t), self.w1(q_j)], dim=2).view(-1, 2 * self.emb_dim)).view(
            mask_i.size())
        miu = torch.exp(miu) * mask_i
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)

        z_j = self.aggre_users_i(torch.sum(miu.unsqueeze(2).expand_as(f_jt) * self.w1(f_jt), 1))
        z_j = F.dropout(z_j, p=0.5, training=self.training)




        q_a = self.item_emb(u_item_pad[:, :, 0])
        mask_u = torch.where(u_item_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        u_item_er = self.rate_emb(u_item_pad[:, :, 1])

        x_ia = self.g_v(torch.cat([q_a, u_item_er], dim=2).view(-1, 2 * self.emb_dim)).view(
            q_a.size())


        p_i = mask_u.unsqueeze(2).expand_as(x_ia) * self.user_emb(uids).unsqueeze(1).expand_as(
            q_a)

        alpha = self.user_items_att(torch.cat([self.w1(x_ia), self.w1(p_i)], dim=2).view(-1, 2 * self.emb_dim)).view(
            mask_u.size())

        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)


        h_iI = self.aggre_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ia) * x_ia, 1))
        h_iI = F.dropout(h_iI, p=0.5, training=self.training)

        z_dot = torch.mul(z_j, h_iI)
        z_cat = torch.cat([z_j, h_iI], dim=1)
        h = self.combine_mlp(torch.cat([z_dot, z_cat], dim=1))

        return h

        return z_j


class GraphRec(nn.Module):


    def __init__(self, num_users, num_items, num_rate_levels, emb_dim=64):
        super(GraphRec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_rate_levels = num_rate_levels
        self.emb_dim = emb_dim
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx=0)
        self.rate_emb = nn.Embedding(int(self.num_rate_levels), self.emb_dim, padding_idx=0)


        self.user_model = _UserModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb)

        self.item_model = _ItemModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb)

        self.rate_pred = nn.Sequential(
            nn.Linear(3 * self.emb_dim, self.emb_dim, bias=True),
            nn.BatchNorm1d(self.emb_dim, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear( self.emb_dim, self.emb_dim//4 ),
            nn.BatchNorm1d(self.emb_dim//4, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.emb_dim//4 , 1)
        )

    def forward(self, uids, iids, u_item_pad, u_user_pad, u_user_item_pad, i_user_pad):


        h = self.user_model(uids, u_item_pad, u_user_pad, u_user_item_pad)
        z = self.item_model(iids, i_user_pad, uids, u_item_pad)


        r_ij_dot = torch.mul(h,z)
        r_ij = torch.cat([h,z],dim=1)



        temp = self.rate_pred(torch.cat([r_ij,r_ij_dot],dim=1))
        return temp
