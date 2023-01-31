import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

class CCS(object):
    def __init__(self, verbose=False, include_bias=True):
        self.includa_bias = include_bias
        self.verbose = verbose


    def add_ones_dimension(self, h):
        if self.includa_bias:
            return np.concatenate([h, np.ones(h.shape[0])[:, None]], axis=-1)
        else:
            return h

    def get_confidence_loss(self, p0, p1):
        """
        Assumes p0 and p1 are each a tensor of probabilities of shape (n,1) or (n,)
        Assumes p0 is close to 1-p1
        Encourages p0 and p1 to be close to 0 or 1 (far from 0.5)
        """
        min_p = torch.min(p0, p1)
        return (min_p**2).mean(0)
        #return (min_p).mean(0)**2  # seems a bit worse
    
    
    def get_similarity_loss(self, p0, p1):
        """
        Assumes p0 and p1 are each a tensor of probabilities of shape (n,1) or (n,)
        Encourages p0 to be close to 1-p1 and vice versa
        """
        return ((p0 - (1-p1))**2).mean(0)
    
    
    def get_loss(self, p0, p1):
        """
        Returns the ConsistencyModel loss for two probabilities each of shape (n,1) or (n,)
        p0 and p1 correspond to the probabilities
        """
        similarity_loss = self.get_similarity_loss(p0, p1)
        confidence_loss = self.get_confidence_loss(p0, p1)
        
        return similarity_loss + confidence_loss
    
    # return the probability tuple (p0, p1)
    def transform(self, data: list, theta_np = None):
        if theta_np is None:
            theta_np = self.best_theta
        z0, z1 = torch.tensor(self.add_ones_dimension(data[0]).dot(theta_np.T)), torch.tensor(self.add_ones_dimension(data[1]).dot(theta_np.T))
        p0, p1 = torch.sigmoid(z0).numpy(), torch.sigmoid(z1).numpy()

        return p0, p1

    # Return the accuracy of (data, label)
    def get_acc(self, theta_np, data: list, label, getloss):
        """
        Computes the accuracy of a given direction theta_np represented as a numpy array
        """
        p0, p1 = self.transform(data, theta_np)
        avg_confidence = 0.5*(p0 + (1-p1))
        
        label = label.reshape(-1)
        predictions = (avg_confidence < 0.5).astype(int)[:, 0]
        acc = (predictions == label).mean()
        if getloss:
            loss = self.get_loss(torch.tensor(p0), torch.tensor(p1)).cpu().detach().item()
            return max(acc, 1 - acc), loss 
        return max(acc, 1 - acc)
    
        
    def train(self):
        """
        Does a single training run of nepochs epochs
        """

        # convert to tensors
        x0 = torch.tensor(self.x0, dtype=torch.float, requires_grad=False, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float, requires_grad=False, device=self.device)
        
        # initialize parameters
        if self.init_theta is None:
            init_theta = np.random.randn(self.d).reshape(1, -1)
            init_theta = init_theta / np.linalg.norm(init_theta)
        else:
            init_theta = self.init_theta
        theta = torch.tensor(init_theta, dtype=torch.float, requires_grad=True, device=self.device)
        
        # set up optimizer
        optimizer = torch.optim.AdamW([theta], lr=self.lr)

        # Start training (full batch)
        for _ in range(self.nepochs):
            
            # project onto theta
            z0, z1 = x0.mm(theta.T), x1.mm(theta.T)

            # sigmoide to get probability            
            p0, p1 = torch.sigmoid(z0), torch.sigmoid(z1)

            # get the corresponding loss
            loss = self.get_loss(p0, p1)

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            #     theta /= torch.norm(theta)
            
            theta_np = theta.cpu().detach().numpy().reshape(1, -1)
            # print("Norm of theta is " + str(np.linalg.norm(theta_np)))
            loss_np = loss.detach().cpu().item()
        
        return theta_np, loss_np

    def validate_data(self, data):
        assert len(data) == 2 and data[0].shape == data[1].shape

    def get_train_loss(self):
        return self.best_loss

    def visualize(self, losses, accs):
        plt.scatter(losses, accs)
        plt.xlabel("Loss")
        plt.ylabel("Accuracy")
        plt.show()

    # seems 50, 20 can significantly reduce overfitting than 1000, 10
    # switch back to 1000 + 10
    def fit(self, data: list, label, nepochs=1000, ntries=10, lr=1e-2, init_theta=None, device="cuda"):
        """
        Does ntries attempts at training, with different random initializations
        """

        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        
        self.device = device
        
        self.init_theta = init_theta
        if self.init_theta is not None:
            self.ntries = 1
    
        if self.verbose:
            print("String fiting data with Prob. nepochs: {}, ntries: {}, lr: {}".format(
                nepochs, ntries, lr
            ))
        # set up the best loss and best theta found so far
        self.best_loss = np.inf
        self.best_theta = self.init_theta

        best_acc = 0.5
        losses, accs = [], []
        self.validate_data(data)

        self.x0 = self.add_ones_dimension(data[0])
        self.x1 = self.add_ones_dimension(data[1])
        self.y = label.reshape(-1)       
        self.d = self.x0.shape[-1]

        for _ in range(self.ntries):
            # train
            theta_np, loss = self.train()
            
            # evaluate
            acc = self.get_acc(theta_np, data, label, getloss = False)
            
            # save
            losses.append(loss)
            accs.append(acc)
            
            # see if it's the best run so far
            if loss < self.best_loss:
                if self.verbose:
                    print("Found a new best theta. New loss: {:.4f}, new acc: {:.4f}".format(loss, acc))
                self.best_theta = theta_np
                self.best_loss = loss
                best_acc = acc
                
        if self.verbose:
            self.visualize(losses, accs)
        
        return self.best_theta, self.best_loss, best_acc

    def score(self, data: list, label, getloss = False):
        self.validate_data(data)
        return self.get_acc(self.best_theta, data, label, getloss)

