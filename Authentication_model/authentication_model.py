import os, glob, random, math, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# ---------- 1. 数据加载 ----------
def load_desktop(root):
    data = defaultdict(list)
    for uid in sorted(os.listdir(root)):
        desk = os.path.join(root, uid, "wild")
        if not os.path.isdir(desk): continue
        for fp in glob.glob(os.path.join(desk, "*.csv")):
            df = pd.read_csv(fp).select_dtypes(include=[np.number])
            for r in df.to_numpy(dtype=np.float32):
                data[uid].append(r)
    dim = next(iter(data.values()))[0].shape[0]
    return data, dim

# ---------- 2. Residual‑MLP 嵌入网络 ----------
class ResidBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w, w),
            nn.BatchNorm1d(w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.BatchNorm1d(w))
    def forward(self, x):
        return nn.GELU()(x + self.net(x))

class EmbedNet(nn.Module):
    def __init__(self, d_in, d_emb=128, width=512, n_res=2):
        super().__init__()
        layers = [
            nn.Linear(d_in, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(0.3)]
        for _ in range(n_res):
            layers.append(ResidBlock(width))
        layers += [nn.Linear(width, d_emb)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return nn.functional.normalize(self.net(x), p=2, dim=1)

# ---------- 3. Semi‑hard Triplet 挖掘 ----------
def semi_hard_triplets(emb_pos, emb_neg, margin=0.3, max_trip=10000):
    a_idx, p_idx, n_idx = [], [], []
    pos = emb_pos.cpu(); neg = emb_neg.cpu()
    d_pp = torch.cdist(pos, pos, p=2)
    d_pn = torch.cdist(pos, neg, p=2)
    for i in range(pos.size(0)):
        j = int(torch.randint(0, pos.size(0)-1, (1,)))
        if j >= i: j += 1
        d_ap = d_pp[i, j].item()
        valid_neg = torch.where((d_pn[i] > d_ap) & (d_pn[i] < d_ap + margin))[0]
        if len(valid_neg) == 0:
            valid_neg = torch.topk(d_pn[i], k=1, largest=False).indices
        k = valid_neg[int(torch.randint(0, len(valid_neg), (1,)))]
        a_idx.append(i); p_idx.append(j); n_idx.append(k.item())
        if len(a_idx) >= max_trip: break
    return torch.tensor(a_idx), torch.tensor(p_idx), torch.tensor(n_idx)

class TripletDataset(Dataset):
    def __init__(self, pos_feats, neg_feats, a_idx, p_idx, n_idx):
        self.pos, self.neg = pos_feats, neg_feats
        self.a, self.p, self.n = a_idx, p_idx, n_idx
    def __len__(self): return len(self.a)
    def __getitem__(self, i):
        return (self.pos[self.a[i]], self.pos[self.p[i]], self.neg[self.n[i]])

# ---------- 4. 评估指标 & 可视化（添加PR曲线、AP指标） ----------

def eval_metrics(dist_pos, dist_neg, plot=False, out_dir=None):
    """
    计算 ACC / EER / AUC / AP，并可选择绘制 & 保存 ROC、PR 曲线及其数值。

    Parameters
    ----------
    dist_pos : 1-D array
        目标用户（正类）的距离分数
    dist_neg : 1-D array
        其它用户（负类）的距离分数
    plot : bool, default False
        是否绘图&保存
    out_dir : str or None
        若给定，将把 png 和 npz 文件保存到此目录
    """
    import os

    y = np.hstack([np.ones_like(dist_pos), np.zeros_like(dist_neg)])
    d = np.hstack([dist_pos, dist_neg])

    # ---------- ROC ----------
    fpr, tpr, thr = roc_curve(y, -d)
    fnr          = 1 - tpr
    idx          = np.argmin(np.abs(fpr - fnr))          # EER 点位置
    eer          = (fpr[idx] + fnr[idx]) / 2
    th           = -thr[idx]                             # 阈值取负号还原
    acc          = ((d <= th) == y).mean()
    auc_val      = auc(fpr, tpr)

    # ---------- PR ----------
    precision_curve, recall_curve, _ = precision_recall_curve(y, -d)
    ap_val = average_precision_score(y, -d)

    # Precision / Recall at EER-threshold
    y_pred          = (d <= th).astype(int)
    tp              = np.sum((y_pred == 1) & (y == 1))
    fp              = np.sum((y_pred == 1) & (y == 0))
    fn              = np.sum((y_pred == 0) & (y == 1))
    precision_at_eer = tp / (tp + fp) if (tp + fp) else 0.0
    recall_at_eer    = tp / (tp + fn) if (tp + fn) else 0.0

    # ----------- 打印 ----------
    print(f"ACC={acc*100:.2f}%  EER={eer*100:.2f}%  "
          f"AUC={auc_val:.4f}  FAR={fpr[idx]*100:.2f}%  FRR={fnr[idx]*100:.2f}%  "
          f"Precision={precision_at_eer*100:.2f}%  Recall={recall_at_eer*100:.2f}%  "
          f"AP={ap_val:.4f}")

    # ----------- 绘图 / 保存 ----------
    if plot:
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            prefix = lambda fn: os.path.join(out_dir, fn)
        else:
            prefix = lambda fn: fn

        # ROC PNG
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.4f}")
        plt.plot([0,1],[0,1],"--",lw=1)
        plt.scatter(fpr[idx], tpr[idx], c='red', s=40,
                    label=f"EER={eer*100:.2f}%")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(prefix("roc_curve.png"), dpi=200)
        plt.close()

        # PR PNG
        plt.figure(figsize=(5,4))
        plt.plot(recall_curve, precision_curve, lw=2, label=f"AP={ap_val:.4f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("PR Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(prefix("pr_curve.png"), dpi=200)
        plt.close()

        # 保存数值
        print('开始保存数值')
        np.savez(prefix("roc_curve_data.npz"),
                 fpr=fpr, tpr=tpr, thr=thr, auc=auc_val)
        np.savez(prefix("pr_curve_data.npz"),
                 recall=recall_curve, precision=precision_curve, ap=ap_val)

    # 与旧接口保持一致
    return eer, auc_val, acc



# ---------- 5. 主流程 ----------
def main(cfg):
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    data, d_in = load_desktop(cfg.root)
    assert cfg.target in data, f"{cfg.target} not found!"
    pos_all = data[cfg.target]

    # ✨ 指定保留的未知用户
    all_users = list(data.keys())
    all_users.remove(cfg.target)
    random.shuffle(all_users)
    # unknown_users = all_users[:4]
    # known_users = all_users[4:]
    # 固定指定未知用户
    unknown_users = ["user18", "user19", "user20", "user21"]
    known_users = [u for u in data.keys() if u not in unknown_users and u != cfg.target]

    print(f"[INFO] 当前轮次选定的未知用户（仅用于测试阶段）：{unknown_users}")
    print(f"[INFO] 当前轮次选定的已知用户为：{known_users}")

    neg_known_all = [s for u in known_users for s in data[u]]
    neg_unknown_all = [s for u in unknown_users for s in data[u]]

    # 划分正样本
    random.shuffle(pos_all); k = math.ceil(len(pos_all)*.7)
    pos_tr, pos_te = pos_all[:k], pos_all[k:]

    # # 划分负样本
    # random.shuffle(neg_known_all)
    # m = math.ceil(len(neg_known_all)*.7)
    # neg_tr = neg_known_all[:m]
    # neg_te = neg_known_all[m:] + neg_unknown_all

    # 划分已知用户负样本
    random.shuffle(neg_known_all)
    m_known = math.ceil(len(neg_known_all) * 0.7)
    neg_tr = neg_known_all[:m_known]  # 已知用户70%用于训练
    neg_known_te = neg_known_all[m_known:]  # 已知用户30%用于测试

    # 划分未知用户负样本
    random.shuffle(neg_unknown_all)
    m_unknown = math.ceil(len(neg_unknown_all) * 0.7)
    neg_unknown_te = neg_unknown_all[m_unknown:]  # 未知用户30%用于测试

    # 最终负样本测试集 = 已知用户30%测试集 + 未知用户30%测试集
    neg_te = neg_known_te + neg_unknown_te

    # 标准化
    scaler = StandardScaler().fit(np.vstack(pos_tr + neg_tr))
    pos_tr = torch.tensor(scaler.transform(pos_tr), dtype=torch.float32)
    neg_tr = torch.tensor(scaler.transform(neg_tr), dtype=torch.float32)
    pos_te = torch.tensor(scaler.transform(pos_te), dtype=torch.float32)
    neg_te = torch.tensor(scaler.transform(neg_te), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = EmbedNet(d_in, cfg.embed_dim, width=cfg.width, n_res=cfg.n_res).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.TripletMarginLoss(margin=cfg.margin, p=2)
    best_eer, patience = 1.0, 0

    print(">>> Start training with semi‑hard mining")
    print(f"\n>>> Start training for user {cfg.target}")
    for epoch in range(1, cfg.epochs+1):
        net.eval()
        with torch.no_grad():
            emb_pos = net(pos_tr.to(device)).cpu()
            emb_neg = net(neg_tr.to(device)).cpu()
        a_idx, p_idx, n_idx = semi_hard_triplets(
            emb_pos, emb_neg, margin=cfg.margin, max_trip=cfg.trip_per*len(pos_tr))
        ds = TripletDataset(pos_tr, neg_tr, a_idx, p_idx, n_idx)
        loader = DataLoader(ds, batch_size=cfg.batch, shuffle=True, drop_last=True)

        net.train(); total = 0.0
        for a,p,n in loader:
            a,p,n = a.to(device),p.to(device),n.to(device)
            opt.zero_grad()
            la,lp,ln = net(a),net(p),net(n)
            loss = loss_fn(la, lp, ln)
            loss.backward(); opt.step()
            total += loss.item()*a.size(0)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{cfg.epochs}  "
              f"Loss={total/len(loader.dataset):.4f}  LR={scheduler.get_last_lr()[0]:.2e}")

        if epoch % 5 == 0:
            net.eval()
            with torch.no_grad():
                center = net(pos_tr.to(device)).mean(0)
                d_pos = torch.norm(net(pos_tr.to(device)) - center, dim=1).cpu().numpy()
                d_neg = torch.norm(net(neg_tr.to(device)) - center, dim=1).cpu().numpy()
            eer, _, _ = eval_metrics(d_pos, d_neg)
            if eer < best_eer - 0.001:
                best_eer, patience = eer, 0
                torch.save(net.state_dict(), "best_model.pth")
                print(f"  >> New best EER {best_eer*100:.2f}%  (model saved)")
            else:
                patience += 1
            if patience >= cfg.es_patience:
                print("Early stopping triggered."); break

    print("\n>>> Evaluation on unseen TEST set")
    print(f"\n>>> Evaluation for user {cfg.target}")
    net.load_state_dict(torch.load("best_model.pth", map_location=device))
    net.eval()
    with torch.no_grad():
        center = net(pos_tr.to(device)).mean(0)
        d_pos_te = torch.norm(net(pos_te.to(device)) - center, dim=1).cpu().numpy()
        d_neg_all = torch.norm(net(neg_te.to(device)) - center, dim=1).cpu().numpy()
        if len(d_neg_all) > len(d_pos_te):
            np.random.seed(0)
            idx = np.random.choice(len(d_neg_all), size=len(d_pos_te), replace=False)
            d_neg_te = d_neg_all[idx]
            print(f"[INFO] 负样本从 {len(d_neg_all)} 下采样为 {len(d_neg_te)} 条以平衡正负样本")
        else:
            d_neg_te = d_neg_all
            print(f"[INFO] 负样本数量较少，未进行下采样")
    # ---- 保存目录：roc_pr_curves/<用户名>/ ----
    save_dir = os.path.join("roc_pr_curves_ourselves", cfg.target)
    os.makedirs(save_dir, exist_ok=True)  # 保证目录存在
    eval_metrics(d_pos_te, d_neg_te, plot=True, out_dir=save_dir)
    # eval_metrics(d_pos_te, d_neg_te, plot=cfg.plot, out_dir=save_dir)
    # eval_metrics(d_pos_te, d_neg_te, plot=cfg.plot)

if __name__ == "__main__":
    target_users = [
        "user1","user2","user3","user4",
        "user5","user6","user7","user8","user9",
        "user10","user11","user12","user13",
        "user14","user15","user16","user17"
    ]

    for user in target_users:
        p = argparse.ArgumentParser()
        p.add_argument("--root", default="../DataProcess/ourselves_features/fusion_feature_wild")
        p.add_argument("--target", default=user, help="目标用户文件夹名")
        p.add_argument("--epochs", type=int, default=150)
        p.add_argument("--batch", type=int, default=256)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--margin", type=float, default=0.02)
        p.add_argument("--embed_dim", type=int, default=128)
        p.add_argument("--width", type=int, default=512)
        p.add_argument("--n_res", type=int, default=1)
        p.add_argument("--trip_per", type=int, default=25)
        p.add_argument("--es_patience", type=int, default=7)
        p.add_argument("--plot", action="store_true", help="保存 ROC 图")
        main(p.parse_args())