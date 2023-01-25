def log_values(
    cost, epoch, batch_id, step, log_likelihood, reinforce_loss, tb_logger, opts
):
    avg_cost = cost.mean().item()

    # Log values to screen
    print(
        "epoch: {}, train_batch_id: {}, avg_cost: {}".format(epoch, batch_id, avg_cost)
    )

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value("avg_cost", avg_cost, step)

        tb_logger.log_value("actor_loss", reinforce_loss.item(), step)
        tb_logger.log_value("nll", -log_likelihood.mean().item(), step)
