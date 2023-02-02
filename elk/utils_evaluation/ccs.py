import numpy as np
import torch
import matplotlib.pyplot as plt
from elk.utils_evaluation.probes import Probe, LinearProbe


class CCS(object):
    def __init__(
        self,
        verbose=False,
        include_bias=True,
        num_epochs=1000,
        num_tries=10,
        learning_rate=1e-2,
        weight_decay=0.01,
        use_lbfgs=False,
        device="cuda",
    ):
        self.include_bias = include_bias
        self.verbose = verbose
        self.num_epochs = num_epochs
        self.num_tries = num_tries
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_lbfgs = use_lbfgs
        self.device = device

    def init_probe(self):
        """
        Initializes the probe of the model.

        Returns:
            Probe: The initialized probe of the model.
        """
        return LinearProbe(self.d, self.include_bias).to(self.device)

    def get_confidence_loss(self, p0, p1):
        """
        Assumes p0 and p1 are each a tensor of probabilities of shape (n,1) or (n,)
        Assumes p0 is close to 1-p1
        Encourages p0 and p1 to be close to 0 or 1 (far from 0.5)
        """
        min_p = torch.min(p0, p1)
        return (min_p ** 2).mean(0)

    def get_consistency_loss(self, p0, p1):
        """
        Assumes p0 and p1 are each a tensor of probabilities of shape (n,1) or (n,)
        Encourages p0 to be close to 1-p1 and vice versa
        """
        return ((p0 - (1 - p1)) ** 2).mean(0)

    def get_loss(self, p0, p1):
        """
        Returns the ConsistencyModel loss for
        two probabilities each of shape (n,1) or (n,)
        p0 and p1 correspond to the probabilities
        """
        consistency_loss = self.get_consistency_loss(p0, p1)
        confidence_loss = self.get_confidence_loss(p0, p1)

        return consistency_loss + confidence_loss

    def transform(self, data: list, probe=None):
        """return the probability tuple (p0, p1)
        Each has shape (n,1)"""
        if probe is None:
            probe = self.best_probe
        p0 = probe(torch.tensor(data[0], device=self.device))
        p1 = probe(torch.tensor(data[1], device=self.device))

        return p0, p1

    # Return the accuracy of (data, label)
    def get_accuracy(self, probe, data: list, label, getloss):
        """
        Computes the accuracy of a probe on a dataset
        """
        with torch.no_grad():
            p0t, p1t = self.transform(data, probe)
        p0, p1 = p0t.cpu().detach().numpy(), p1t.cpu().detach().numpy()
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

        probe = self.init_probe()
        if self.use_lbfgs:
            loss = self.train_loop_lbfgs(x0, x1, probe)
        else:
            loss = self.train_loop_full_batch(x0, x1, probe)

        loss_np = loss.detach().cpu().item()

        return probe, loss_np

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
        self.best_probe = None

        best_acc = 0.5
        losses, accuracies = [], []
        self.validate_data(data)

        self.x0 = data[0]
        self.x1 = data[1]
        self.y = label.reshape(-1)
        self.d = self.x0.shape[-1]

        for _ in range(self.num_tries):
            probe, loss = self.single_train()

            accuracy = self.get_accuracy(probe, data, label, getloss=False)

            losses.append(loss)
            accuracies.append(accuracy)

            if loss < self.best_loss:
                if self.verbose:
                    print(
                        "Found a new best theta. New loss: {:.4f}, \
                        new acc: {:.4f}".format(
                            loss, accuracy
                        )
                    )
                self.best_probe = probe
                self.best_loss = loss
                best_acc = accuracy

        if self.verbose:
            self.visualize(losses, accuracies)

        return self.best_probe, self.best_loss, best_acc

    def score(self, data: list, label, getloss=False):
        self.validate_data(data)
        return self.get_accuracy(self.best_probe, data, label, getloss)

    def train_loop_full_batch(self, x0, x1, probe):
        """
        Performs a full batch training loop. Modifies the probe in place.
        """
        optimizer = torch.optim.AdamW(
            probe.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Start training (full batch)
        for _ in range(self.num_epochs):

            p0, p1 = probe(x0), probe(x1)

            # get the corresponding loss
            loss = self.get_loss(p0, p1)

            loss += probe.normalize()

            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # with torch.no_grad():
            #     theta /= torch.norm(theta)

        return loss

    def train_loop_lbfgs(self, x0, x1, probe):
        """
        Performs a lbfgs training loop. Modifies the probe in place.
        """

        l2 = self.weight_decay

        # set up optimizer
        optimizer = torch.optim.LBFGS(
            probe.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=self.num_epochs,
            tolerance_change=torch.finfo(x0.dtype).eps,
            tolerance_grad=torch.finfo(x0.dtype).eps,
        )

        def closure():
            optimizer.zero_grad()

            p0, p1 = probe(x0), probe(x1)

            # get the corresponding loss
            loss = self.get_loss(p0, p1)

            # compute l2 loss
            for param in probe.parameters():
                loss += l2 * torch.norm(param) ** 2 / 2

            loss += probe.normalize()

            # update the parameters
            loss.backward()

            # with torch.no_grad():
            #     theta /= torch.norm(theta)

            return loss

        optimizer.step(closure)
        return closure()
