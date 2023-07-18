import torch
import os 


def save_checkpoint(state, save_path, filename, fold):
    torch.save(state, os.path.join(save_path, f'model_last_{fold}.pth'))
        
def save_dataset_checkpoint(state, is_best, save_path, filename, fold, dataset):
    if is_best:
        torch.save(state, os.path.join(save_path, f'model_best_{dataset}_{fold}.pth'))
            
            
def load_checkpoint(flow_generator, img_generator, emotion_discriminator, discriminator,\
                    optimizer_1, optimizer_2, disc_opt_1, disc_opt_2,\
                    lr_scheduler_G1, lr_scheduler_G2,\
                    filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val = -1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        
        flow_generator.load_state_dict(checkpoint['flow_gen_state_dict'])
        img_generator.load_state_dict(checkpoint['img_gen_state_dict'])
        emotion_discriminator.load_state_dict(checkpoint['emotion_disc_state_dict'])
        discriminator.load_state_dict(checkpoint['disc_state_dict'])
        
        optimizer_1.load_state_dict(checkpoint['flow_gen_optimizer'])
        optimizer_2.load_state_dict(checkpoint['img_gen_optimizer'])
        disc_opt_1.load_state_dict(checkpoint['disc_optimizer'])
        disc_opt_2.load_state_dict(checkpoint['emotion_disc_optimizer'])
        
        lr_scheduler_G1.load_state_dict(checkpoint['flow_gen_scheduler'])
        lr_scheduler_G2.load_state_dict(checkpoint['img_gen_scheduler'])
        try:
            best_val=checkpoint['best_val']
        except:
            best_val=-1
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))



    ## update the optimizers
    for state in optimizer_1.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    for state in optimizer_2.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    for state in disc_opt_1.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    for state in disc_opt_2.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return  flow_generator, img_generator, emotion_discriminator, discriminator,\
    optimizer_1, optimizer_2, disc_opt_1, disc_opt_2,\
    lr_scheduler_G1, lr_scheduler_G2,\
    start_epoch, best_val


def get_best_val(filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        #print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        best_val=checkpoint['best_val']

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return  best_val