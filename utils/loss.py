
import torch.nn.functional as F

def geometry_loss(pred_joints, true_joints):
    # pred_joints: [B, T, 18], reshape为 (B, T, 6, 3)
    pred = pred_joints.view(-1, pred_joints.size(1), 6, 3)
    true = true_joints.view(-1, true_joints.size(1), 6, 3)

    def pair_distance(joints, a, b):
        return ((joints[:,:,a] - joints[:,:,b])**2).sum(-1).sqrt()

    # 肩-肘距离：0-1左臂，3-4右臂；肘-腕：1-2左臂，4-5右臂
    l_upper = pair_distance(pred, 0, 1)
    r_upper = pair_distance(pred, 3, 4)
    l_lower = pair_distance(pred, 1, 2)
    r_lower = pair_distance(pred, 4, 5)

    ldcl = F.mse_loss(l_upper, r_upper) + F.mse_loss(l_lower, r_lower)
    return ldcl