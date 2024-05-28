import torch
from torch import nn
import matplotlib.pyplot as plt


##### util functions for adverarial training #####
# source: https://adversarial-ml-tutorial.org/adversarial_training/ 

def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.05, alpha=1e-3, num_iter=20, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()
    
##################################################


##### visualizations #####

def plot_losses(losses):
    plt.figure(figsize=(10, 4))
    for label, loss_hist in losses.items():
        plt.plot(range(len(loss_hist)), loss_hist, label=label, alpha=0.4)
    plt.legend()
    plt.show()

##########################