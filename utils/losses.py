import torch
import torch.nn as nn
import torchvision


categorical_crossentropy = nn.CrossEntropyLoss()
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixelwise = nn.L1Loss()

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow, 2, 1).mean()


def L1_charbonnier_loss(input_flow, target_flow, alpha=0.4, epsilon=0.01):
    L1 = criterion_pixelwise(input_flow, target_flow)
    return torch.pow(L1 + epsilon, alpha).mean()
    

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
            
        self.blocks = nn.ModuleList(blocks).to('cuda')
        self.transform = nn.functional.interpolate
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))
        self.resize = resize

    def forward(self, input_, target):
        if input_.shape[1] != 3:
            input_ = input_.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input_ = (input_-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input_ = self.transform(input_, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input_
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += nn.functional.l1_loss(x, y)
        return loss


def motion_warper_loss(gen_img, tar_img, lambda_mae, lambda_perc):
    loss = lambda_mae * criterion_pixelwise(tar_img, gen_img) + lambda_perc * perceptual_loss(gen_img, tar_img) 
    return loss
    
    
def motion_normalizer_loss(gen_flow, tar_flow, disc_output, true_labels, disc_labels_output,\
                           lambda_l1, lambda_epe):
    
    """motion normalizer loss
    
    generator_loss = gan_loss + reconstruction_loss
    
    1). gan_loss = .5 * discriminator_1_loss + .5 * discriminator_2_loss
    
    2). reconstruction_loss = lambda_wing * wing_loss + lambda_mse * mse_loss + lambda_ep * endpoint_loss + lambda_identity * identity_loss
    """
        
    gan_loss = .5 * criterion_GAN(disc_output, torch.ones_like(disc_output)) +\
               .5 * categorical_crossentropy(disc_labels_output, true_labels.long())
      
    ## 2. RECONSTRUCTION LOSS
    epe_loss = EPE(gen_flow, tar_flow)
    l1_loss = L1_charbonnier_loss(gen_flow, tar_flow)
    loss = lambda_l1 * l1_loss + lambda_epe * epe_loss
        
    ## GENERATOR LOSS = GAN LOSS + RECONSTRUCTION LOSS
    total_gen_loss = gan_loss + loss
    
    return total_gen_loss, gan_loss