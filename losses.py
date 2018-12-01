import torch


def dice_score(input, target, is_onehot_form=True, keep_background=False):
    """
    assume input shape: [Batch, ..., channel, H, W]
    is_onehot_form control whether it's onehot_form or mask form
    converting from mask form to onehot form may take a lot of memory and time so this function allow 2nd option
    :param input:
    :param target:
    :param smooth:
    :param keep_background:
    :return:
    """
    smooth = 1e-8
    if is_onehot_form:
        iflat = input.view(list(input.shape[:-2]) + [-1])
        tflat = target.view(list(target.shape[:-2]) + [-1])

        if not keep_background:
            iflat = iflat[..., 1:, :]
            tflat = tflat[..., 1:, :]

        intersection = (iflat * tflat).sum(dim=-1)

        return ((2. * intersection + smooth) /
                (iflat.sum(dim=-1) + tflat.sum(dim=-1) + smooth))
    else:
        iflat = input.view(list(input.shape[:-2]) + [-1])
        tflat = target.view(list(target.shape[:-2]) + [-1])

        cls_i = list(torch.unique(input).data.cpu().numpy())
        cls_t = list(torch.unique(target).data.cpu().numpy())
        cls = cls_i + list(set(cls_t) - set(cls_i))
        cls = sorted(cls)
        scores = []
        for j in cls:
            if not keep_background and j == 0:
                continue
            iflat_mask = (iflat == int(j)).type(torch.cuda.FloatTensor)
            tflat_mask = (tflat == int(j)).type(torch.cuda.FloatTensor)
            intersection = (iflat_mask * tflat_mask).sum(dim=-1, keepdim=True)
            score = ((2. * intersection + smooth) /
                     (iflat_mask.sum(dim=-1, keepdim=True) + tflat_mask.sum(dim=-1, keepdim=True) + smooth))
            scores.append(score)
        scores = torch.cat(scores, dim=-1)
        return scores


def dice_score_inside_bboxes(inputs, targets, bboxes, is_onehot_form=True, keep_background=False):
    """
    assume input shape: [Batch, ..., channel, H, W]
    is_onehot_form control whether it's onehot_form or mask form
    converting from mask form to onehot form may take a lot of memory and time so this function allow 2nd option
    :param input:
    :param target:
    :param smooth:
    :param keep_background:
    :return:
    """
    scores = []
    for j, bbox in enumerate(bboxes):
        row_min, row_max, col_min, col_max = bbox
        if is_onehot_form:
            input = inputs[j:j+1, ..., row_min:row_max, col_min:col_max].clone()
            target = targets[j:j+1, ..., row_min:row_max, col_min:col_max].clone()
        else:
            input = inputs[j:j+1, row_min:row_max, col_min:col_max].clone()
            target = targets[j:j+1, row_min:row_max, col_min:col_max].clone()
        scores.append(dice_score(input, target, is_onehot_form, keep_background))
    scores = torch.cat(scores)
    return scores


def dice_loss(input, target, is_onehot_form=True, keep_background=False):
    return 1 - dice_score(input, target, is_onehot_form, keep_background)
