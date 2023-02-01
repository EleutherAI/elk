import numpy as np
import time
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt


class myReduction:
    def __init__(
        self, method, n_components, print_more=False, svd_solver="full"
    ) -> None:
        self.n_components = n_components
        self.method = method
        assert method in ["PCA", "UMAP"], NotImplementedError(
            "Only support PCA and UMAP to project data."
        )
        self.print_more = print_more
        self.num_feature = None
        if n_components != -1:
            if self.method == "PCA":
                self.model = PCA(n_components=n_components, svd_solver=svd_solver)
            elif self.method == "UMAP":
                self.model = umap.UMAP(n_components=n_components)

    def fit(self, data):
        self.num_feature = data.shape[1]
        if self.n_components == -1:
            if self.print_more:
                print("n_components = -1, will return identity")

        else:
            if self.method == "UMAP":  # for UMAP, explicitly centralize the data
                data = data - np.mean(data, axis=0)
            self.model.fit(data)
            if self.method == "PCA":  # for PCA, explicitly set mean to None
                self.model.mean_ = None
                if self.print_more:
                    print("Set the mean of PCA model to `None`.")
            if self.print_more:
                if self.method == "PCA":
                    print(
                        "PCA fit data. dim = {} and #data = {}, var is {}".format(
                            self.n_components,
                            data.shape,
                            sum(self.model.explained_variance_ratio_),
                        )
                    )
                else:
                    print(
                        "UMAP fit data. dim = {} and #data = {}.".format(
                            self.n_components, data.shape
                        )
                    )

    def getDirection(self):
        # return the component with shape (n_components, n_features)
        if self.n_components == -1:
            return np.eye(self.num_feature)
        else:
            return self.model.components_

    def transform(self, data):
        if self.n_components == -1:
            return data
        return self.model.transform(data)

    def __getattr__(self, __name):
        if __name == "n_components":
            return self.n_components
        return getattr(self.model, __name)


def getSingleLoss(x, verbose=False):
    # x: shape (n, 1)
    x1 = x[x < 0]
    x2 = x[x >= 0]

    if verbose:
        print(
            "var(x1) = {}, var(x2) = {}, var(x) = {}".format(
                x1.var(), x2.var(), x.var()
            )
        )
    return (x1.var() + x2.var()) / x.var()


def getLoss(z, weights, verbose=False):
    # weighted loss according to `weights`
    return sum([u * getSingleLoss(x, verbose) for u, x in zip(weights, z)])


def get_all_data(data_dict):
    all_data, all_labels = [], []
    for dataset in data_dict.keys():
        raw_data = np.concatenate([w[0] for w in data_dict[dataset]], axis=0)
        label = np.concatenate([w[1] for w in data_dict[dataset]])

        all_data.append(raw_data)
        all_labels.append(label)
    all_data, all_labels = np.concatenate(all_data), np.concatenate(all_labels)

    hs0, hs1 = (
        all_data[:, : all_data.shape[-1] // 2],
        all_data[:, all_data.shape[-1] // 2 :],
    )

    return hs0, hs1, all_labels


class ConsistencyMethod(object):
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
        # return (min_p).mean(0)**2  # seems a bit worse

    def get_similarity_loss(self, p0, p1):
        """
        Assumes p0 and p1 are each a tensor of probabilities of shape (n,1) or (n,)
        Encourages p0 to be close to 1-p1 and vice versa
        """
        return ((p0 - (1 - p1)) ** 2).mean(0)

    def get_loss(self, p0, p1):
        """
        Returns ConsistencyModel loss for two probabilities each of shape (n,1) or (n,)
        p0 and p1 correspond to the probabilities
        """
        similarity_loss = self.get_similarity_loss(p0, p1)
        confidence_loss = self.get_confidence_loss(p0, p1)

        return similarity_loss + confidence_loss

    # return the probability tuple (p0, p1)
    def transform(self, data: list, theta_np=None):
        if theta_np is None:
            theta_np = self.best_theta
        z0, z1 = torch.tensor(
            self.add_ones_dimension(data[0]).dot(theta_np.T)
        ), torch.tensor(self.add_ones_dimension(data[1]).dot(theta_np.T))
        p0, p1 = torch.sigmoid(z0).numpy(), torch.sigmoid(z1).numpy()

        return p0, p1

    # Return the accuracy of (data, label)
    def get_acc(self, theta_np, data: list, label, getloss):
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

    def train(self):
        """
        Does a single training run of nepochs epochs
        """

        # convert to tensors
        x0 = torch.tensor(
            self.x0, dtype=torch.float, requires_grad=False, device=self.device
        )
        x1 = torch.tensor(
            self.x1, dtype=torch.float, requires_grad=False, device=self.device
        )

        # initialize parameters
        if self.init_theta is None:
            init_theta = np.random.randn(self.d).reshape(1, -1)
            init_theta = init_theta / np.linalg.norm(init_theta)
        else:
            init_theta = self.init_theta
        theta = torch.tensor(
            init_theta, dtype=torch.float, requires_grad=True, device=self.device
        )

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
    def fit(
        self,
        data: list,
        label,
        nepochs=1000,
        ntries=10,
        lr=1e-2,
        init_theta=None,
        device="cuda",
    ):
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
            print(
                "String fiting data with Prob. nepochs: {}, ntries: {}, lr: {}".format(
                    nepochs, ntries, lr
                )
            )
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
            acc = self.get_acc(theta_np, data, label, getloss=False)

            # save
            losses.append(loss)
            accs.append(acc)

            # see if it's the best run so far
            if loss < self.best_loss:
                if self.verbose:
                    print(f"New best theta. New loss: {loss:.4f}, new acc: {acc:.4f}")
                self.best_theta = theta_np
                self.best_loss = loss
                best_acc = acc

        if self.verbose:
            self.visualize(losses, accs)

        return self.best_theta, self.best_loss, best_acc

    def score(self, data: list, label, getloss=False):
        self.validate_data(data)
        return self.get_acc(self.best_theta, data, label, getloss)


class myClassifyModel(LogisticRegression):
    def __init__(self, method, print_more=False):
        assert method in [
            "TPC",
            "LR",
            "BSS",
            "KMeans",
        ], "currently only support method to be `TPC`, `LR`, 'KMeans` and `BSS`!"
        self.method = method
        super(myClassifyModel, self).__init__(max_iter=10000, n_jobs=1, C=0.1)
        self.print_more = print_more

    def set_params(self, coef, bias):
        self.classes_ = np.array([0, 1])
        self.intercept_ = bias
        self.coef_ = coef

    def get_train_loss(self):
        assert self.method == "BSS", NotImplementedError(
            "`get_train_loss` supported only when method is `BSS`."
        )
        return self.loss

    def fit(
        self,
        data,
        label,
        times=20,
        use_scheduler=False,
        weights=None,
        lr=1e-1,
        epochs=20,
        device="cuda",
    ):
        if self.method == "LR":
            super().fit(data, label)
            if self.print_more:
                print(
                    "fitting to {} data, acc is {}".format(
                        len(label), self.score(data, label)
                    )
                )

        elif self.method == "TPC":
            h_dim = data.shape[1]
            assert (
                h_dim == 1
            ), f"When `avg` mode is used, #hidden_dim should be 1, but it's {h_dim}"

            self.avg = 0.0
            self.sign = 1

            debias = (data > 0).reshape(label.shape).astype(int)
            if np.sum(debias == label) / label.shape[0] < 0.5:
                self.sign = -1

            # set to model parameters
            self.set_params(np.array(self.sign).reshape(1, 1), -self.sign * self.avg)

        elif self.method == "KMeans":
            self.model = KMeans(n_clusters=2)
            self.model.fit(data)
            if self.print_more:
                print(
                    "fitting to {} data, acc is {}".format(
                        len(label), self.score(data, label)
                    )
                )

        elif self.method == "BSS":  # in this case, `data` will be a list
            assert (
                type(data) == list
            ), "When using BSS mode, data should be a list instead of {}".format(
                type(data)
            )

            x = [torch.tensor(w, device=device) for w in data]
            dim = data[0].shape[1]  # hidden dimension

            if weights is None:
                weights = [1 / len(x) for _ in range(len(x))]
            else:
                assert type(weights) == list and len(weights) == len(
                    x
                ), "Length of `weights` mismatches length of `data`."
                weights = [w / sum(weights) for w in weights]  # normalize

            sample_weight = [
                u / w.shape[0] for u, w in zip(weights, data) for _ in range(w.shape[0])
            ]

            minloss = 1.0
            final_coef = np.random.randn(dim).reshape(1, -1)
            final_bias = 0.0
            for _ in range(times):
                init_theta = np.random.randn(dim).reshape(1, -1)
                init_theta /= np.linalg.norm(init_theta)

                theta = torch.tensor(
                    init_theta, dtype=torch.float, requires_grad=True, device=device
                )
                optimizer = torch.optim.AdamW([theta], lr=lr)
                if use_scheduler:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, factor=0.5, verbose=self.print_more, min_lr=1e-6
                    )

                for epoch in range(epochs):
                    z = [w @ theta.T for w in x]

                    loss = getLoss(z, weights)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        theta /= torch.norm(theta)

                    if use_scheduler:
                        scheduler.step(loss)

                    if ((epoch + 1) % 50 == 0 and self.print_more) or epoch in [
                        0,
                        epochs - 1,
                    ]:
                        theta_np = (
                            theta.cpu().detach().numpy().reshape(1, -1)
                        )  # same as coef

                        projected, gth = np.concatenate(
                            [w @ theta_np.T for w in data]
                        ).reshape(-1), np.concatenate(label).reshape(-1)

                        self.avg = 0.0
                        self.sign = 1
                        debias = (projected > 0).reshape(gth.shape).astype(int)
                        if np.sum(debias == gth) / gth.shape[0] < 0.5:
                            self.sign = -1

                        # set to model parameters
                        self.set_params(self.sign * theta_np, -self.sign * self.avg)

                        # TODO: this was in the original code but is never accessed.
                        # Is this a bug? (N.B.)
                        # acc = self.score(
                        #     np.concatenate(data, axis=0),
                        #     np.concatenate(label),
                        #     sample_weight,
                        # )

                # check whether this time gives a lower loss
                with torch.no_grad():
                    z = [w @ theta.T for w in x]
                    # if weights is None:
                    loss = sum([getSingleLoss(w, False) for w in z]) / len(z)
                    loss = loss.detach().cpu().item()
                    if loss < minloss:
                        if self.print_more:
                            print(
                                "update params, acc is {:.2f}, old loss is {:.4f}, new"
                                " loss is {:.4f}".format(
                                    100
                                    * self.score(
                                        np.concatenate(data, axis=0),
                                        np.concatenate(label),
                                        sample_weight,
                                    ),
                                    minloss,
                                    loss,
                                )
                            )
                        minloss = loss
                        final_coef = self.coef_
                        final_bias = self.intercept_

            # update loss
            self.loss = minloss
            self.set_params(final_coef, final_bias)

    def score(self, data, label, getloss=False, sample_weight=None):
        if self.method == "KMeans":
            prediction = self.model.predict(data)
            acc = max(np.mean(prediction == label), np.mean(1 - prediction == label))
            if getloss:
                return acc, 0.0
            return acc
        else:
            if sample_weight is not None:
                acc = super().score(data, label, sample_weight)
            else:
                acc = super().score(data, label)
            if getloss:
                if self.method == "BSS":
                    loss = getSingleLoss(data @ self.coef_.T + self.intercept_)
                else:
                    loss = 0.0
                return acc, loss
            return acc


def getConcat(data_list, axis=0):
    sub_list = [w for w in data_list if w is not None]
    if sub_list == []:
        return None
    return np.concatenate(sub_list, axis=axis)


def getPair(target_dict, data_dict, permutation_dict, projection_model, split="train"):
    split_idx = 0 if split == "train" else 1
    lis = []
    for key, prompt_lis in target_dict.items():
        for idx in prompt_lis:
            lis.append(
                [
                    projection_model.transform(
                        data_dict[key][idx][0][permutation_dict[key][split_idx]]
                    ),
                    data_dict[key][idx][1][permutation_dict[key][split_idx]],
                ]
            )  # each is a data & label paird, selecting the corresponding split

    data, label = getConcat([w[0] for w in lis]), getConcat([w[1] for w in lis])

    return data, label


print(
    "------ Func: mainResults ------\n## Input = (data_dict, permutation_dict,"
    " projection_dict, test_dict, train_on)test, n_components, method, print_more ="
    " False, learn_dict = {}) ##\n    data_dict: Dict of hidden states loaded from"
    " `getDic()`.\n    permutation_dict: Dict of permutation loaded from `getDic()`.\n "
    "   projection_dict: Key is set_name, each value is a list of prompt_idx that is"
    " used to do projection.\n    test_dict: Test indices, results in this list will be"
    " return.\n    projection_method: The method you use to do projection. Can be `PCA`"
    " or `UMAP`.\n    n_components: The dimension you want to reduce to. -1 means no"
    " projection will be implemented.\n    projection_only: Default is false. When set"
    " to true, will directly return the `projection_model`, and `res`, `classify_model`"
    " will be None.\n    classification_method: Method used to predict, including LR,"
    " TPC and BSS. Default is BSS.\n    print_more: Whether to print more.\n   "
    " learn_dict: A dict to specify the learning parameters for torch. See class"
    " `classify_model` for details.\n## Output = (res, projection_model,"
    " classify_model) ##\n    res: a dict (key, acc_list). Key is the name of dataset,"
    " and acc_list is a list with each accuracy corresponds to one prompt of set"
    " `key`.\n    projection_model & classify_model: the model after training. Can be"
    " used to do any further prediction.\n"
)


def mainResults(
    # dict of hidden states, key is set_name, each value is a list w/ len = #promp_idx
    data_dict,
    # dict of permutation, key is set_name, contain 2 array indicating train and test
    # split
    permutation_dict,
    # projection dict, key is set_name, each value is a list of prompt_idx being used
    # to do projection
    projection_dict,
    test_dict,  # test indices, results in this list will be return
    # When set to True, will immediate return after we train the projection_model.
    # res and classify_model will be None.
    projection_method="PCA",
    # The dimension you want to reduce to. -1 means no projection will be implemented.
    n_components=2,
    projection_only=False,
    classification_method="BSS",  # can be LR, TPC and BSS
    print_more=False,
    learn_dict={},
):
    start = time.time()
    if print_more:
        print(
            (
                "Projection method: {} (n_con = {}) in {}\n"
                "Classification method: {} in: {}"
            ).format(
                projection_method,
                n_components,
                projection_dict,
                classification_method,
                test_dict,
            )
        )

    # use all data (not split) to do the PCA
    proj_states = getConcat(
        [
            getConcat([data_dict[key][w][0] for w in lis])
            for key, lis in projection_dict.items()
        ]
    )
    projection_model = myReduction(
        method=projection_method, n_components=n_components, print_more=print_more
    )
    projection_model.fit(proj_states)

    if projection_only:
        return None, projection_model, None

    if classification_method == "Prob":
        classify_model = ConsistencyMethod(verbose=print_more)
        data_array, label = getPair(
            data_dict=data_dict,
            permutation_dict=permutation_dict,
            projection_model=projection_model,
            target_dict=projection_dict,
        )
        assert len(data_array.shape) == 2
        data = [
            data_array[:, : data_array.shape[1] // 2],
            data_array[:, data_array.shape[1] // 2 :],
        ]
        classify_model.fit(data=data, label=label, **learn_dict)

    elif classification_method == "BSS":
        lis = [
            getPair(
                data_dict=data_dict,
                permutation_dict=permutation_dict,
                projection_model=projection_model,
                target_dict={key: [idx]},
            )
            for key, indices in projection_dict.items()
            for idx in indices
        ]

        weights = [
            1 / len(indices) for indices in projection_dict.values() for _ in indices
        ]

        classify_model = myClassifyModel(
            method=classification_method, print_more=print_more
        )
        classify_model.fit(
            [w[0] for w in lis], [w[1] for w in lis], weights=weights, **learn_dict
        )

    else:
        classify_model = myClassifyModel(
            method=classification_method, print_more=print_more
        )
        classify_model.fit(
            *getPair(
                data_dict=data_dict,
                permutation_dict=permutation_dict,
                projection_model=projection_model,
                target_dict=projection_dict,
            )
        )

    res, lss = {}, {}
    for key, lis in test_dict.items():
        res[key], lss[key] = [], []
        for prompt_idx in lis:
            dic = {key: [prompt_idx]}
            data, label = getPair(
                data_dict=data_dict,
                permutation_dict=permutation_dict,
                projection_model=projection_model,
                target_dict=dic,
                split="test",
            )
            if classification_method == "Prob":
                data = [data[:, : data.shape[1] // 2], data[:, data.shape[1] // 2 :]]
            acc, loss = classify_model.score(data, label, getloss=True)
            res[key].append(acc)
            lss[key].append(loss)

    duration = time.time() - start
    if print_more:
        print("mainResults finished, duration: {}s".format(duration))
    return res, lss, projection_model, classify_model


print(
    "------ Func: printAcc ------\n## Input = (input_dict, verbose) ##\n    input_dict:"
    " The dict generated by `mainResults`.\n    verbose: Whether to print dataset level"
    " accuracy.\n## Output ##\n    Directly print the accuracy and return the global"
    " level accuracy.\n"
)


def printAcc(input_dic, verbose=1):
    if type(input_dic) != dict:
        print(input_dic)
        return np.mean(input_dic)
    if verbose >= 2:
        for key in input_dic.keys():
            print(
                "Test on {}, avg acc is {:.2f}, best is {:.2f}, std is {:.2f}".format(
                    key,
                    100 * np.mean(input_dic[key]),
                    100 * np.max(input_dic[key]),
                    100 * np.std(input_dic[key]),
                )
            )
    global_acc = np.mean([100 * np.mean(w) for w in input_dic.values()])
    global_std = np.mean([100 * np.std(w) for w in input_dic.values()])
    if verbose >= 1:
        print("## Global accuracy: {:.2f}, std.: {:.2f}".format(global_acc, global_std))
    return global_acc
