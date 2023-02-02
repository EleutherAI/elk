import numpy as np
import torch
import matplotlib.pyplot as plt


class CCS(object):
    def __init__(
        self,
        verbose=False,
        include_bias=True,
        num_epochs=1000,
        num_tries=10,
        learning_rate=1e-2,
        device="cuda",
    ):
        self.include_bias = include_bias
        self.verbose = verbose
        self.num_epochs = num_epochs
        self.num_tries = num_tries
        self.learning_rate = learning_rate
        self.device = device

    def init_parameters(self):
        """
        Initializes the parameters of the model.

        Returns:
            numpy.ndarray: The initialized parameters of the model.
        """
        init_theta = np.random.randn(self.d).reshape(1, -1)
        init_theta = init_theta / np.linalg.norm(init_theta)
        return init_theta

    def add_ones_dimension(self, x):
        """
        Adds an additional dimension of ones to the input x,
        if include_bias is True, else returns the original input x.

        Parameters:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The input array x with an additional dimension of ones,
            if include_bias is True. Otherwise, returns the original input x.
            otherwise it simply returns the original input x without any modifications.
        """
        if self.include_bias:
            # by adding a dimension of ones to the input array,
            # the bias term can be easily included in the calculation without having to modify the input data
            ones = np.ones(x.shape[0])[:, None]
            return np.concatenate([x, ones], axis=-1)
        else:
            return x

    def get_confidence_loss(self, p0, p1):
        """
        Assumes p0 and p1 are each a tensor of probabilities of shape (n,1) or (n,)
        Assumes p0 is close to 1-p1
        Encourages p0 and p1 to be close to 0 or 1 (far from 0.5)
        """
        min_p = torch.min(p0, p1)
        return (min_p**2).mean(0)

    def get_consistency_loss(self, p0, p1):
        """
        Assumes p0 and p1 are each a tensor of probabilities of shape (n,1) or (n,)
        Encourages p0 to be close to 1-p1 and vice versa
        """
        return ((p0 - (1 - p1)) ** 2).mean(0)

    def get_loss(self, p0, p1):
        """
        Returns the ConsistencyModel loss for two probabilities each of shape (n,1) or (n,)
        p0 and p1 correspond to the probabilities
        """
        consistency_loss = self.get_consistency_loss(p0, p1)
        confidence_loss = self.get_confidence_loss(p0, p1)

        return consistency_loss + confidence_loss

    # return the probability tuple (p0, p1)

    def transform(self, data: list, theta_np=None):
        if theta_np is None:
            theta_np = self.best_theta
        z0 = torch.tensor(self.add_ones_dimension(data[0]).dot(theta_np.T))
        z1 = torch.tensor(self.add_ones_dimension(data[1]).dot(theta_np.T))

        p0 = torch.sigmoid(z0).numpy()
        p1 = torch.sigmoid(z1).numpy()

        return p0, p1

    # Return the accuracy of (data, label)
    def get_accuracy(self, theta_np, data: list, label, getloss):
        """
        Computes the accuracy of a given direction theta_np represented as a numpy array
        """
        p0, p1 = self.transform(data, theta_np)
        avg_confidence = 0.5 * (p0 + (1 - p1))

        label = label.reshape(-1)
        predictions = (avg_confidence < 0.5).astype(int)[:, 0]
        acc = (predictions == label).mean()
        if getloss:
            loss = (
                self.get_loss(torch.tensor(p0), torch.tensor(p1)).cpu().detach().item()
            )
            return max(acc, 1 - acc), loss
        return max(acc, 1 - acc)

    def single_train(self):
        """
        Does a single training run of num_epochs epochs
        """

        x0 = torch.tensor(
            self.x0, dtype=torch.float, requires_grad=False, device=self.device
        )
        x1 = torch.tensor(
            self.x1, dtype=torch.float, requires_grad=False, device=self.device
        )

        theta = self.init_parameters()
        theta = torch.tensor(
            theta, dtype=torch.float, requires_grad=True, device=self.device
        )

        optimizer = torch.optim.AdamW([theta], lr=self.learning_rate)

        # Start training (full batch)
        for _ in range(self.num_epochs):

            # project onto theta
            z0, z1 = x0.mm(theta.T), x1.mm(theta.T)

            # sigmoid to get probability
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
    def fit(self, data: list, label):
        if self.verbose:
            print(
                f"String fiting data with Prob. num_epochs: {self.num_epochs},"
                f" num_tries: {self.num_tries}, learning_rate: {self.learning_rate}"
            )
        # set up the best loss and best theta found so far
        self.best_loss = np.inf
        self.best_theta = None

        best_acc = 0.5
        losses, accuracies = [], []
        self.validate_data(data)

        self.x0 = self.add_ones_dimension(data[0])
        self.x1 = self.add_ones_dimension(data[1])
        self.y = label.reshape(-1)
        self.d = self.x0.shape[-1]

        for _ in range(self.num_tries):
            theta_np, loss = self.single_train()

            accuracy = self.get_accuracy(theta_np, data, label, getloss=False)

            losses.append(loss)
            accuracies.append(accuracy)

            if loss < self.best_loss:
                if self.verbose:
                    print(
                        "Found a new best theta. New loss: {:.4f}, new acc: {:.4f}".format(
                            loss, accuracy
                        )
                    )
                self.best_theta = theta_np
                self.best_loss = loss
                best_acc = accuracy

        if self.verbose:
            self.visualize(losses, accuracies)

        return self.best_theta, self.best_loss, best_acc

    def score(self, data: list, label, getloss=False):
        self.validate_data(data)
        return self.get_accuracy(self.best_theta, data, label, getloss)
