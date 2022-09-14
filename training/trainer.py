import numpy as np
import torch

import time


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        batch_size,
        loaders,
        loss_fn,
        device,
        scheduler=None,
        eval_fns=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loaders = loaders
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.device = device

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for batch in self.loaders["train"]:
            # put batch on device
            for k, v in batch.items():
                if k == "obs" or k == "next_obs":
                    batch[k] = (
                        torch.cat([batch[k][k2] for k2 in batch[k].keys()], dim=-1)
                        .to(self.device)
                        .to(dtype=torch.float32)
                    )
                batch[k] = batch[k].to(self.device)
            train_loss = self.train_step(batch)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs["time/training"] = time.time() - train_start

        # eval_start = time.time()

        # self.model.eval()
        # for eval_fn in self.eval_fns:
        #     outputs = eval_fn(self.model)
        #     for k, v in outputs.items():
        #         logs[f"evaluation/{k}"] = v

        logs["time/total"] = time.time() - self.start_time
        # logs["time/evaluation"] = time.time() - eval_start
        logs["training/train_loss_mean"] = np.mean(train_losses)
        logs["training/train_loss_std"] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        return logs

    def train_step(self, batch):
        actions = batch["actions"]
        B, T, D = actions.shape
        states = batch["obs"]
        attention_mask = torch.ones((B, T)).to(states.device)
        timesteps = batch["timesteps"]

        state_target, action_target = (
            torch.clone(states),
            torch.clone(actions),
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            masks=None,
            attention_mask=attention_mask,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds,
            action_preds,
            state_target[:, 1:],
            action_target,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
