manual_seed = 1234 # reproduce experiemnt
random_sample = True # whether to sample the dataset with random sampler
img_h = 48 # the height of the input image to network
img_w = 432 # the width of the input image to network
nh = 256 # size of the lstm hidden state
nc = 3
dealwith_lossnan = False # whether to replace all nan/inf in gradients to zero

# hardware
cuda = True # enables cuda
multi_gpu = False # whether to use multi gpu
ngpu = 1 # number of GPUs to use. Do remember to set multi_gpu to True!
workers = 10 # number of data loading workers

# training process
n_val_disp = 10 # number of samples to display when val the model

# finetune
batchSize = 32 # input batch size
lr = 0.0001 # learning rate for Critic, not used by adadealta
beta1 = 0.5 # beta1 for adam. default=0.5
adam = True # whether to use adam (default is rmsprop)
adadelta = False # whether to use adadelta (default is rmsprop)
