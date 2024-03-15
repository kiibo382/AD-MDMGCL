import os
from itertools import chain
from typing import Callable, Dict, Tuple, Union

import dgl
import numpy as np
import pyaml
import torch
from torch.distributions import Bernoulli
from torch.optim.lr_scheduler import *  # For loading optimizer specified in config
from torch.utils.data import DataLoader
from commons.utils import move_to_device

from trainer.trainer import Trainer
from models import *  # do not remove
from trainer.byol_wrapper import BYOLwrapper
from trainer.lr_schedulers import WarmUpWrapper  # do not remove
import copy


class SelfSupervisedAdversarialTrainer(Trainer):
    def __init__(
        self,
        model,
        model3d,
        args,
        metrics: Dict[str, Callable],
        main_metric: str,
        device: torch.device,
        tensorboard_functions: Dict[str, Callable],
        optim=None,
        main_metric_goal: str = "min",
        loss_func=torch.nn.MSELoss,
        scheduler_step_per_batch: bool = True,
        view_learner=None,
        view_optim=None,
        **kwargs,
    ):
        self.model3d = model3d.to(
            device
        )  # move to device before loading optim params in super class
        super(SelfSupervisedAdversarialTrainer, self).__init__(
            model,
            args,
            metrics,
            main_metric,
            device,
            tensorboard_functions,
            optim,
            main_metric_goal,
            loss_func,
            scheduler_step_per_batch,
        )
        self.view_learner = view_learner.to(device)
        self.view_optim = view_optim(self.view_learner.parameters(), **args.optimizer_params)
        self.initialize_view_learner_scheduler()
        self.aug_type = args.aug_type
        self.lambda_reg = args.loss_params["lambda_reg"] if "lambda_reg" in args.loss_params else 0.0

        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.model3d.load_state_dict(checkpoint["model3d_state_dict"])
            self.view_learner.load_state_dict(checkpoint["view_learner_state_dict"])
            self.view_optim.load_state_dict(
                checkpoint["view_learner_optimizer_state_dict"]
            )
            if (
                self.view_learner_lr_scheduler is not None
                and checkpoint["view_learner_scheduler_state_dict"] is not None
            ):
                self.view_learner_lr_scheduler.load_state_dict(checkpoint["view_learner_scheduler_state_dict"])

    def forward_pass(self, batch):
        info2d, info3d, *snorm_n = tuple(batch)
        info2d_cp = copy.deepcopy(info2d)

        logits = self.view_learner(*info2d)

        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + logits) / temperature
        weight = torch.sigmoid(gate_inputs).squeeze()

        reg = None
        view2d = self.model(
            *info2d, *snorm_n
        )  # foward the rest of the batch to the model
        view3d = self.model3d(*info3d)
        if self.aug_type == 'node_drop':
            view2d_aug = self.model(
                *info2d_cp, edge_weight=None, node_weight=weight
            )  # foward the rest of the batch to the model
            if isinstance(info2d[0], dgl.DGLGraph):
                info2d[0].ndata["batch_weight"] = weight
                reg = []
                for unbatch in dgl.unbatch(info2d[0]):
                    reg.append(unbatch.ndata["batch_weight"].sum()/unbatch.number_of_nodes())
                reg = torch.stack(reg)
                reg = reg.mean()
        elif self.aug_type == 'edge_drop':
            view2d_aug = self.model(
                *info2d_cp, edge_weight=weight, node_weight=None
            )
            if isinstance(info2d[0], dgl.DGLGraph):
                info2d[0].edata["batch_weight"] = weight
                reg = []
                for unbatch in dgl.unbatch(info2d[0]):
                    reg.append(unbatch.edata["batch_weight"].sum()/unbatch.number_of_edges())
                reg = torch.stack(reg)
                reg = reg.mean()
        loss = self.loss_func(
            view2d,
            view2d_aug,
            view3d,
            nodes_per_graph=info2d[0].batch_num_nodes()
            if isinstance(info2d[0], dgl.DGLGraph)
            else None,
        )

        return loss, view2d, view3d, reg

    def process_batch(self, batch, optim, view_optim):
        loss, view2d, view3d, reg = self.forward_pass(batch)
        if optim is not None:  # run backpropagation if an optimizer is provided
            loss.backward()
            optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            optim.zero_grad()
            self.optim_steps += 1
        if view_optim is not None:
            (-loss+(self.lambda_reg * reg)).backward()
            view_optim.step()
            view_optim.zero_grad()
        return loss, view2d.detach(), view3d.detach()

    def predict(
        self,
        data_loader: DataLoader,
        epoch: int,
        optim: torch.optim.Optimizer = None,
        return_predictions: bool = False,
    ) -> Union[
        Dict, Tuple[float, Union[torch.Tensor, None], Union[torch.Tensor, None]]
    ]:
        total_metrics = {
            k: 0
            for k in list(self.metrics.keys())
            + [
                type(self.loss_func).__name__,
                "mean_pred",
                "std_pred",
                "mean_targets",
                "std_targets",
            ]
        }
        epoch_targets = torch.tensor([]).to(self.device)
        epoch_predictions = torch.tensor([]).to(self.device)
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            # ic(self.optim.param_groups)
            batch_copy = copy.deepcopy(batch)
            batch = move_to_device(list(batch), self.device)
            if optim is not None:
                self.view_learner.train()
                self.view_learner.zero_grad()
                self.model.eval()
                self.model3d.eval()
                self.process_batch(batch, optim=None, view_optim=self.view_optim)
                optim.zero_grad()

            batch = move_to_device(list(batch_copy), self.device)

            self.model.train()
            self.model3d.train()
            self.model.zero_grad()
            self.model3d.zero_grad()
            self.view_learner.eval()
            loss, predictions, targets = self.process_batch(
                batch, optim=optim, view_optim=None
            )

            with torch.no_grad():
                if self.optim_steps % self.args.log_iterations == 0 and optim != None:
                    metrics = self.evaluate_metrics(predictions, targets)
                    metrics[type(self.loss_func).__name__] = loss.item()
                    self.run_tensorboard_functions(
                        predictions, targets, step=self.optim_steps, data_split="train"
                    )
                    self.tensorboard_log(
                        metrics, data_split="train", step=self.optim_steps, epoch=epoch
                    )
                    print(
                        "[Epoch %d; Iter %5d/%5d] %s: loss: %.7f"
                        % (epoch, i + 1, len(data_loader), "train", loss.item())
                    )
                    if self.main_metric == "InfoNCE":
                        # Mutual Information = log(number of negative samples) - NTXent
                        print(
                            "Mutual Information: ",
                            np.log(self.args.batch_size - 1) - loss.item(),
                        )
                if (
                    optim == None and self.val_per_batch
                ):  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics_results = self.evaluate_metrics(
                        predictions, targets, val=True
                    )
                    metrics_results[type(self.loss_func).__name__] = loss.item()
                    if i == 0 and epoch in self.args.models_to_save:
                        self.run_tensorboard_functions(
                            predictions,
                            targets,
                            step=self.optim_steps,
                            data_split="val",
                        )
                    for key, value in metrics_results.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch:
                    epoch_loss += loss.item()
                    epoch_targets = torch.cat((targets, epoch_targets), 0)
                    epoch_predictions = torch.cat((predictions, epoch_predictions), 0)

        if optim == None:
            if self.val_per_batch:
                total_metrics = {
                    k: v / len(data_loader) for k, v in total_metrics.items()
                }
            else:
                total_metrics = self.evaluate_metrics(
                    epoch_predictions, epoch_targets, val=True
                )
                total_metrics[type(self.loss_func).__name__] = epoch_loss / len(
                    data_loader
                )
            return total_metrics

    def evaluate_metrics(self, z2d, z3d, batch=None, val=False) -> Dict[str, float]:
        metric_results = {}
        metric_results[f"mean_pred"] = torch.mean(z2d).item()
        metric_results[f"std_pred"] = torch.std(z2d).item()
        metric_results[f"mean_targets"] = torch.mean(z3d).item()
        metric_results[f"std_targets"] = torch.std(z3d).item()
        if "Local" in type(self.loss_func).__name__ and batch != None:
            node_indices = torch.cumsum(batch[0].batch_num_nodes(), dim=0)
            pos_mask = torch.zeros((len(z2d), len(z3d)), device=z2d.device)
            for graph_idx in range(1, len(node_indices)):
                pos_mask[
                    node_indices[graph_idx - 1] : node_indices[graph_idx], graph_idx
                ] = 1.0
            pos_mask[0 : node_indices[0], 0] = 1
            for key, metric in self.metrics.items():
                if not hasattr(metric, "val_only") or val:
                    metric_results[key] = metric(z2d, z3d, pos_mask).item()
        else:
            for key, metric in self.metrics.items():
                if not hasattr(metric, "val_only") or val:
                    metric_results[key] = metric(z2d, z3d).item()
        return metric_results

    def run_per_epoch_evaluations(self, data_loader):
        print("fitting linear probe")
        representations = []
        targets = []
        for batch in data_loader:
            batch = [element.to(self.device) for element in batch]
            loss, view2d, view3d = self.process_batch(batch, optim=None)
            representations.append(view2d)
            targets.append(batch[-1])
            if len(representations) * len(view2d) >= self.args.linear_probing_samples:
                break
        representations = torch.cat(representations, dim=0)
        targets = torch.cat(targets, dim=0)
        if len(representations) >= representations.shape[-1]:
            X, _ = torch.lstsq(targets, representations)
            X, _ = torch.lstsq(targets, representations)
            sol = X[: representations.shape[-1]]
            pred = representations @ sol
            mean_absolute_error = (pred - targets).abs().mean()
            self.writer.add_scalar(
                "linear_probe_mae", mean_absolute_error.item(), self.optim_steps
            )
        else:
            raise ValueError(
                f"We have less linear_probing_samples {len(representations)} than the metric dimension {representations.shape[-1]}. Linear probing cannot be used."
            )

        print("finish fitting linear probe")

    def initialize_optimizer(self, optim):
        normal_params = [
            v
            for k, v in chain(
                self.model.named_parameters(), self.model3d.named_parameters()
            )
            if not "batch_norm" in k
        ]
        batch_norm_params = [
            v
            for k, v in chain(
                self.model.named_parameters(), self.model3d.named_parameters()
            )
            if "batch_norm" in k
        ]

        self.optim = optim(
            [
                {"params": batch_norm_params, "weight_decay": 0},
                {"params": normal_params},
            ],
            **self.args.optimizer_params,
        )

    def step_schedulers(self, metrics=None):
        try:
            self.lr_scheduler.step(metrics=metrics)
        except:
            self.lr_scheduler.step()
        try:
            self.view_learner_lr_scheduler.step(metrics=metrics)
        except:
            self.view_learner_lr_scheduler.step()

    def initialize_view_learner_scheduler(self):
        if (
            self.args.lr_scheduler
        ):  # Needs "from torch.optim.lr_scheduler import *" to work
            self.view_learner_lr_scheduler = globals()[self.args.lr_scheduler](
                self.view_optim, **self.args.lr_scheduler_params
            )
        else:
            self.view_learner_lr_scheduler = None

    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save(
            {
                "epoch": epoch,
                "best_val_score": self.best_val_score,
                "optim_steps": self.optim_steps,
                "model_state_dict": self.model.state_dict(),
                "model3d_state_dict": self.model3d.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "view_learner_state_dict": self.view_learner.state_dict(),
                "view_learner_optimizer_state_dict": self.view_optim.state_dict(),
                "scheduler_state_dict": None
                if self.lr_scheduler == None
                else self.lr_scheduler.state_dict(),
                "view_learner_scheduler_state_dict": None
                if self.view_learner_lr_scheduler is None
                else self.view_learner_lr_scheduler.state_dict(),
            },
            os.path.join(self.writer.log_dir, checkpoint_name),
        )
