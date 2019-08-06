import torch
import torch.nn as nn
import torch.nn.functional as F

def trilinear_interpolation(frames, offsets):
    """
    Interpolating the frames by trilinear, return the sampled pixels
    :param frames: shape [b, c, d, h, w], in which, b->batch, c->channel, d->color channel
    :param offsets: per-pixel 3D offsets, shape [b, N*3, h, w], N is the number of sampled pixels
    :return: shape [b, N, d, h, w]
    """
    b, c, d, h, w = frames.size()
    N = offsets.size(1) // 3
    device = frames.device.type
    # reshape it to suit the shape
    offsets = offsets.view(b, 3*N, 1, h, w).expand(b, 3*N, d, h, w)
    # results
    tri_res = torch.zeros(b, N, d, h, w, device=device)
    h_pos, w_pos = torch.meshgrid(
        torch.arange(start=0, end=h), torch.arange(start=0, end=w)
    )
    h_pos = h_pos.view(1, 1, 1, h, w).expand(b, N, d, h, w).long().to(device)
    w_pos = w_pos.view(1, 1, 1, h, w).expand(b, N, d, h, w).long().to(device)
    # default 0, the middle frame is selected, thus, adding N_pos to offsets
    N_pos = torch.Tensor([c//2]).view(1, 1, 1, 1, 1).expand(b, N, d, h, w).long().to(device)
    # the wanted positions of sampled pixels
    ib = torch.arange(b).view(b, 1, 1, 1, 1).long()
    id = torch.arange(d).view(1, 1, d, 1, 1).long()

    # find the corners of a cubic
    # ceil is 1-floor, to address the problem that floor(0)=0, ceil(0)=0
    floor, ceil = torch.floor, lambda x: torch.floor(x)+1
    f_set = (
        (floor, floor, floor),
        (floor, floor, ceil),
        (floor, ceil, floor),
        (floor, ceil, ceil),
        (ceil, floor, floor),
        (ceil, floor, ceil),
        (ceil, ceil, floor),
        (ceil, ceil, ceil),
    )
    for fN, fh, fw in f_set:
        f_N_pos = fN(offsets[:, 0::3, ...])
        f_h_pos = fh(offsets[:, 1::3, ...])
        f_w_pos = fw(offsets[:, 2::3, ...])
        tri_res += _select_by_index(frames, ib, f_N_pos.long() + N_pos, id, f_h_pos.long() + h_pos, f_w_pos.long() + w_pos) * \
                   (1 - torch.abs(f_N_pos - offsets[:, 0::3, ...])) * (1 - torch.abs(f_h_pos - offsets[:, 1::3, ...])) * \
                   (1 - torch.abs(f_w_pos - offsets[:, 2::3, ...]))
    return tri_res

def _select_by_index(tensor, ib, ic, id, ih, iw):
    """
    :param tensor: [b, c, d, h, w]
    :param iN: [b, N, d, h, w]
    :param ih: [b, N, d, h, w]
    :param iw: as the above
    :return: [b, N, d, h, w]
    """
    b, c, d, h, w = tensor.size()
    # if the position is outside the tensor, make the mask and set them to zero
    mask_outside = ((0 <= ic).int() + (ic < c).int() + (0 <= ih).int() + (ih < h).int() + (0 <= iw).int() + (iw < w).int()) != 6
    ic[mask_outside] = 0
    ih[mask_outside] = 0
    iw[mask_outside] = 0

    res = tensor[ib, ic, id, ih, iw]
    res[mask_outside] = 0
    return res

def convolution_3D(samples, kernels):
    """
    convolved the samples with kernels at corresponding positions
    :param samples: shape [b, N, d, h, w]
    :param kernels: shape [b, N, h, w]
    :return: shape [b, d, h, w]
    """
    assert len(samples.size()) == 5
    assert len(kernels.size()) == 4
    b, N, d, h, w = samples.size()
    kernels = kernels.view(b, N, 1, h, w).expand(b, N, d, h, w)
    return torch.sum(samples*kernels, dim=1, keepdim=False)

if __name__ == '__main__':
    # trilinear_interpolation(torch.arange(180).reshape(2, 5, 2, 3, 3).float(), 0*torch.ones(2, 3, 3, 3))
    # print(torch.arange(180).reshape(2, 5, 2, 3, 3).float())
    unfold = convolution_3D(torch.arange(81).reshape(1, 3, 3, 3, 3).float(), torch.ones(1,3,3,3))
    print(unfold)
