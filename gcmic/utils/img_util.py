import os
import base64
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torch


def resize_padding(im, desired_size, mode="RGBA", background_color=(0, 0, 0)):
    """
    Args:
        im (pillow image object): the image to be resized
        desired_size (int): the side length of resized image
        mode (string): image mode for creating PIL Image object

    Returns:
        pillow image object: resized image
    """
    # compute the new size
    old_size = im.size
    if max(old_size) == 0:
        return None
    ratio = float(desired_size) / max(old_size)
    new_size = tuple(int(x * ratio) for x in old_size)
    w, h = new_size
    # if h <= 2 or w <= 2:
    #     return None

    # create a new image and paste the resized on it
    if mode == "L":
        im = im.resize(new_size, Image.NEAREST)
        new_im = Image.new(mode, (desired_size, desired_size))
    elif mode in ["RGB", "RGBA"]:
        im = im.resize(new_size)
        new_im = Image.new(mode, (desired_size, desired_size), color=background_color)
    new_im.paste(im,
                ((desired_size - new_size[0]) // 2,
                (desired_size - new_size[1]) // 2))
    return new_im


def color_tranfer(source, target, clip=True, preserve_paper=False):
    """
    https://github.com/IGLICT/IBSR_jittor/blob/main/code/ColorTransfer.py
    clip = true and preserve_paper = False will get better results
    """
    def rgb2xyz(rgb): # rgb from [0,1]
        mask = (rgb > .04045).float()
        rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)
        x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
        y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
        z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
        out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]), dim=1)
        return out
    
    def xyz2lab(xyz):
        sc = torch.tensor((0.95047, 1., 1.08883))[None,:,None,None]
        xyz_scale = xyz / sc
        mask = (xyz_scale > .008856).float()
        xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)
        L = 116.*xyz_int[:,1,:,:] - 16.
        a = 500.*(xyz_int[:,0,:,:] - xyz_int[:,1,:,:])
        b = 200.*(xyz_int[:,1,:,:] - xyz_int[:,2,:,:])
        out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]), dim=1)
        return out
    
    def rgb2lab(rgb):
        lab = xyz2lab(rgb2xyz(rgb))
        l_rs = (lab[:,[0],:,:] - 50) / 100
        ab_rs = lab[:,1:,:,:] / 100
        out = torch.cat((l_rs, ab_rs), dim=1)
        return out
    
    def lab2xyz(lab):
        y_int = (lab[:,0,:,:]+16.)/116.
        x_int = (lab[:,1,:,:]/500.) + y_int
        z_int = y_int - (lab[:,2,:,:]/200.)
        z_int = torch.max(torch.tensor((0,)).type_as(lab), z_int)
        out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]), dim=1)
        mask = (out > .2068966).float()
        out = (out**3.) * mask + (out - 16./116.)/7.787 * (1-mask)
        sc = torch.tensor((0.95047, 1., 1.08883)).type_as(lab)[None,:,None,None]
        out = out*sc
        return out
    
    def xyz2rgb(xyz):
        r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
        g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
        b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]
        rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
        rgb = torch.max(rgb, torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs
        mask = (rgb > .0031308).float()
        rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)
        return rgb
    
    def lab2rgb(lab_rs):
        l = lab_rs[:,[0],:,:]*100 + 50
        ab = lab_rs[:,1:,:,:]*100
        lab = torch.cat((l,ab), dim=1)
        out = xyz2rgb(lab2xyz(lab))
        return out
    
    source = source * 255
    target = target * 255

    source = rgb2lab(source)
    target = rgb2lab(target)
    
    # source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    # target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    # source = torch.from_numpy(source).permute(2, 0, 1).unsqueeze(0)
    # target = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)

    MeanSrc = source.mean(dim=(2, 3))
    StdSrc = source.std(dim=(2, 3))

    MeanTar = target.mean(dim=(2, 3))
    StdTar = target.std(dim=(2, 3))

    target -= MeanTar.unsqueeze(-1).unsqueeze(-1) 

    if preserve_paper:
        target =  (StdTar/StdSrc).unsqueeze(-1).unsqueeze(-1) * target
    else:
        target =  (StdSrc/StdTar).unsqueeze(-1).unsqueeze(-1) * target

    target += MeanSrc.unsqueeze(-1).unsqueeze(-1)
    
    target = lab2rgb(target)
    
    if clip:
        transfers = torch.clamp(target, 0, 255)
        transfers = transfers / 255
    else:
        bmin = target.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        bmax = target.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        transfers = (target - bmin) / (bmax - bmin)
        
    # transfers = cv2.cvtColor(transfers.squeeze(0).permute(1,2,0).numpy().astype("uint8"), cv2.COLOR_LAB2BGR)
    
    return transfers


def get_largest_contour(im):
    """Get largest contour of an image

    Args:
        im (H*W numpy array): grayscale image or binary image. OpenCV expects that the foreground objects to detect are in white with the background in black.

    Returns:
        N*1*2 numpy array: contour point set
    """
    items = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # use length 2 tuple, takle different version of OpenCV
    contours = items[0] if len(items) == 2 else items[1]
    if len(contours) == 0:
        return None
    largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    if largest_contour.shape[:2] == (1, 1):
        return None
    return largest_contour


if __name__ == "__main__":
    from torchvision import transforms
    from torchvision.utils import save_image
    src_img = transforms.ToTensor()(Image.open("/path_to_pix3d/img/chair/0011.png").convert("RGB"))
    tgt_img = transforms.ToTensor()(Image.open("/path_to_pix3d/img/chair/0019.png").convert("RGB"))
    
    src_img = src_img.unsqueeze(0)
    tgt_img = tgt_img.unsqueeze(0)
    
    transfer_img = color_tranfer(src_img, tgt_img)
    save_image(transfer_img[0], "./test.png")