import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn import L1Loss
import numpy as np
import pickle5 as pickle
import random

from src.utils import move_to
from src.models.devformer import set_decode_type


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print("Validating...")

    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print(
        "Validation overall avg_cost: {} +- {}".format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))
        )
    )

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    # model.eval()
    set_decode_type(model, "greedy")

    def eval_model_bat(bat):

        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))

        return cost.data.cpu()

    def eval_model_bat_perm_inv(bat):
        np.random.seed(3)
        # sample unsupervised action from fixed copy model
        with torch.no_grad():
            _, log_likelihood_target, pi = model(
                move_to(bat, opts.device), return_pi=True
            )

        # rand perm_aug
        perm = np.random.permutation(20)

        pi = pi[:, perm]

        _, log_likelihood_perm = model(move_to(bat, opts.device), action=pi)

        PA_loss = L1Loss()

        loss_ap = PA_loss(
            torch.exp(log_likelihood_target), torch.exp(log_likelihood_perm)
        )
        return loss_ap.item()

    return torch.cat(
        [
            eval_model_bat(bat)
            for bat in tqdm(
                DataLoader(dataset, batch_size=opts.eval_batch_size),
                disable=opts.no_progress_bar,
            )
        ],
        0,
    )


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm
            if max_norm > 0
            else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    return grad_norms, grad_norms_clipped


def train_epoch(
    model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, opts
):
    print(
        "Start train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]["lr"], opts.run_name
        )
    )
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if opts.EE:
        if opts.wandb:
            import wandb

            wandb.init(project="offline-tsp")
            wandb.run.name = "ours-{}".format(opts.seed)
    else:
        if opts.wandb:
            import wandb

            wandb.init(project="offline-tsp")
            wandb.run.name = "Am-{}".format(opts.seed)
    np.random.rand(1)

    # For IL and Semi-supervised IL, guiding problem-action must be provided from opts.train_dataset
    # if opts.training_mode == "IL" or opts.training_mode == "IL_Perm":
    # Generate new training data for each epoch

    if opts.training_mode == "IL":
        with open(opts.guiding_action, "rb") as f:
            data1 = pickle.load(f)
        # Data Pre-processing
        if opts.problem == "tsp":
            sol_list = []

            for i in range(100):
                sol_list.append(data1[0][i][1])

            solution = np.array(sol_list)
            action = torch.tensor(solution).long().cuda()
        else:
            a = np.array(data1)
            action = torch.tensor(a).long().cuda()

        training_dataset = baseline.wrap_dataset(
            problem.make_dataset(
                size=opts.graph_size,
                num_samples=action.shape[0],
                filename=opts.train_dataset,
                distribution=opts.data_distribution,
            )
        )
    else:
        training_dataset = baseline.wrap_dataset(
            problem.make_dataset(
                size=opts.graph_size,
                num_samples=opts.epoch_size,
                filename=opts.train_dataset,
                distribution=opts.data_distribution,
            )
        )
    training_dataloader = DataLoader(
        training_dataset, batch_size=opts.batch_size, num_workers=1
    )

    # # For RL, we can randomly generate training dataset
    # if opts.training_mode == "RL":
    #     training_dataset = baseline.wrap_dataset(problem.make_dataset(
    #         size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    #     training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(
        tqdm(training_dataloader, disable=opts.no_progress_bar)
    ):

        if opts.training_mode == "IL":
            train_batch(
                model,
                optimizer,
                baseline,
                batch,
                opts,
                action[(batch_id) * opts.batch_size : (batch_id + 1) * opts.batch_size],
            )
        else:
            train_batch(model, optimizer, baseline, batch, opts, None)
        step += 1

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )

    if (
        opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0
    ) or epoch == opts.n_epochs - 1:
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict(),
            },
            os.path.join(opts.save_dir, "epoch-{}.pt".format(epoch)),
        )
        avg_reward = validate(model, val_dataset, opts)

        if opts.wandb:
            wandb.log({"score": -avg_reward})

    # IL and Semi-supervised IL no need baseline
    if opts.training_mode == "RL":
        baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


def symmetric_action(action, opts):
    if opts.problem == "dpp":

        # random ordering of placement
        perm = np.random.permutation(opts.K)
        action = action[:, perm]
    elif opts.problem == "tsp":
        tsp_len = action.shape[1]

        # random traslation of TSP: shifted sequence of TSP is same TSP solution
        i = random.randint(0, tsp_len - 1)

        # random flipping TSP sequences: reverse sequence is same TSP solution
        # flip = random.randint(0,1)
        # if flip == 0:

        action = torch.cat([action[:, i:], action[:, :i]], dim=1)
        # else:
        #     # reverse of sequence
        #     action = action[:,::-1]
        #     action = torch.cat([action[:,i:],action[:,:i]],dim=1)
    return action


def action_aug(action, opts):
    augmented_action = action
    for i in range(opts.N_aug - 1):
        action = torch.cat([augmented_action, symmetric_action(action, opts)], dim=0)

    return action


def train_batch(model, optimizer, baseline, batch, opts, action):

    x, bl_val = baseline.unwrap_batch(batch)

    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Semi-supervised IL
    if opts.training_mode == "IL":
        # action= guiding action

        if opts.EE:
            x = x.repeat(opts.N_aug, 1, 1)
            action = action_aug(action, opts)

        _, log_likelihood_IL = model(x, action=action)

        reinforce_loss = (-log_likelihood_IL).mean()

        # sample unsupervised action from fixed copy model , no grad=fixed policy generate x
        with torch.no_grad():
            _, log_likelihood_target, pi = model(x, return_pi=True)

        # rand perm_aug
        # perm = np.random.permutation(opts.K)

        # pi = pi[:,perm]

        if opts.SE:
            pi = symmetric_action(pi, opts)

            _, log_likelihood_perm = model(x, action=pi)

            PA_loss = L1Loss()

            loss_ap = opts.lamb * PA_loss(
                torch.exp(log_likelihood_target), torch.exp(log_likelihood_perm)
            )

            # clipping heuristic for L_AP
            if loss_ap > 10:
                loss_ap = 10

            loss = reinforce_loss + loss_ap
        else:
            loss = reinforce_loss

    # Reinforcement Learning
    if opts.training_mode == "RL":
        # Evaluate model, get costs and log probabilities
        cost, log_likelihood_IL = model(x)

        # Evaluate baseline, get baseline loss if any (only for critic)
        bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

        # Calculate loss
        reinforce_loss = ((cost - bl_val) * log_likelihood_IL).mean()

        loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()

    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
