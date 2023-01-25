import torch
from torch import nn
import math
from torch.nn import DataParallel

from src.utils.tensor_functions import compute_in_batches
from src.utils.beam_search import CachedLookup
from src.utils.functions import sample_many
from src.models.graph_encoder import GraphAttentionEncoder


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class DevFormerFixed:
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    """

    _fields = (
        "node_embeddings",
        "context_node_projected",
        "glimpse_key",
        "glimpse_val",
        "logit_key",
    )

    def __init__(
        self,
        node_embeddings,
        context_node_projected,
        glimpse_key,
        glimpse_val,
        logit_key,
    ):
        self.node_embeddings = node_embeddings
        self.context_node_projected = context_node_projected
        self.glimpse_key = glimpse_key
        self.glimpse_val = glimpse_val
        self.logit_key = logit_key

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return DevFormerFixed(
                self.node_embeddings[key],
                self.context_node_projected[key],
                self.glimpse_key[:, key],  # dim 0 are the heads
                self.glimpse_val[:, key],  # dim 0 are the heads
                self.logit_key[key],
            )
        else:
            raise TypeError("Invalid index type")


class DevFormer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        problem,
        n_encode_layers=2,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        normalization="batch",
        num_decap=20,
        input_dim=3,
        n_heads=8,
        checkpoint_encoder=False,
        shrink_size=None,
    ):
        super(DevFormer, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.num_decap = num_decap

        # Problem specific context parameters (placeholder and step context dimension)

        step_context_dim = embedding_dim  # Embedding of previeous selected node
        node_dim = input_dim  # [x, y, {is_probing, is_keepout}]

        # Learned input symbols for first action
        self.W_placeholder = nn.Parameter(torch.Tensor(embedding_dim))
        self.W_placeholder.data.uniform_(
            -1, 1
        )  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.prob_embed = nn.Linear(1, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization,
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_prob_MLP = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.project_step_MLP = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.project_node_embeddings = nn.Linear(
            embedding_dim, 3 * embedding_dim, bias=False
        )
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_fixed_context2 = nn.Linear(
            embedding_dim, embedding_dim, bias=False
        )
        self.project_prob_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_prob_context2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(
            step_context_dim, embedding_dim, bias=False
        )
        self.project_step_context2 = nn.Linear(
            step_context_dim, embedding_dim, bias=False
        )
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def PPE(self, input):

        coordinates = input[:, :, :-1]
        probing = input[self.probing].unsqueeze(1)
        probing_coord = probing[:, :, :-1]

        distances = (coordinates - probing_coord).norm(p=2, dim=2).unsqueeze(2)
        embedding = self.prob_embed(distances)
        return embedding

    def forward(self, input, action=None, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        self.probing_point = torch.where(input[:, :, 2] == 2)[1].clone().view(-1, 1)
        self.probing = input[:, :, 2] == 2
        self.keep_out = input[:, :, 2] == 1

        # coordinate embedding
        coordinate_embedding = self._init_embed(input)

        # novel port-based positional encoding
        probing_positional_embedding = self.PPE(input)
        embeddings, _ = self.embedder(
            coordinate_embedding + probing_positional_embedding
        )

        _log_p, pi = self._inner(input, embeddings, action)

        if action == None:

            if return_pi:
                cost = 0
                ll = self._calc_log_likelihood(_log_p, pi, None)
            else:
                cost, mask = self.problem.get_costs(input, pi)
                ll = self._calc_log_likelihood(_log_p, pi, mask)

        else:
            ll = self._calc_log_likelihood(_log_p, action, None)
            cost = 0

        # cost, mask = self.problem.get_costs(input, pi, self.raw_pdn,self.z_init_list)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        # ll = self._calc_log_likelihood(_log_p, pi, mask)

        if return_pi:
            return cost, ll, pi

        # np.save('ours_attention_solution_R_15_15_K_40', pi.cpu().numpy())
        # np.save('ours_attetnion_rewards_R_15_15_K_40', cost.cpu().numpy())
        # print(pi)
        # print(cost)
        return cost, ll

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(
        self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096
    ):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(
                fixed[b.ids], b.state, k=expand_size, normalize=normalize
            ),
            max_calc_batch_size,
            beam,
            n=beam.size(),
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(
            flat_action.size(-1), out=flat_action.new()
        ) // ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (
            log_p > -1000
        ).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        return self.init_embed(input)

    def _inner(self, input, embeddings, action=None, probing=None, keep_out=None):

        # in the case that only inner is called by eval.py function
        #######################################
        if probing is not None:
            self.probing = probing
        if keep_out is not None:
            self.keep_out = keep_out
        #######################################

        outputs = []
        sequences = []

        state = self.problem.make_state(input)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        batch_size = state.ids.size(0)

        # Perform decoding steps
        i = 0
        for i in range(self.num_decap):

            log_p, mask = self._get_log_p(
                fixed, state, probing=self.probing, keep_out=self.keep_out
            )

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            if action == None:
                selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])
            else:

                selected = action[:, i]

            state = state.update(selected)

            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(
                input[0], pi
            ),  # Don't need embeddings as input to get_costs
            (
                input,
                self.embedder(self._init_embed(input))[0],
            ),  # Pack input with embeddings (additional input)
            batch_rep,
            iter_rep,
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(
                1, selected.unsqueeze(-1)
            ).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print("Sampled bad values, resampling!")
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        batch_size = graph_embed.shape[0]
        prob_embed = embeddings.gather(
            1, self.probing_point[:, :, None].expand(batch_size, 1, embeddings.size(-1))
        ).view(batch_size, 1, -1)

        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        # fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        prob_context = self.project_prob_MLP(prob_embed)

        # fixed_context_final = self.project_fixed_context2(fixed_context+prob_context)

        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous(),
        )
        return DevFormerFixed(
            embeddings, prob_context, *fixed_attention_node_data
        )

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(
                log_p.size(0), 1
            )[:, None, :],
        )

    def _get_log_p(self, fixed, state, probing=None, keep_out=None, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + self.project_step_MLP(
            self._get_parallel_step_context(fixed.node_embeddings, state)
        )

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # expands like mask: [batch, 1, problem_size]
        probing = probing.view(probing.shape[0], 1, -1)
        keep_out = keep_out.view(keep_out.shape[0], 1, -1)

        mask = ~(~mask * ~probing * ~keep_out)

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(
            query, glimpse_K, glimpse_V, logit_K, mask
        )

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()

        if (
            num_steps == 1
        ):  # We need to special case if we have only 1 step, may be the first or not
            if state.i.item() == 0:
                # First and only step, ignore prev_a (this is a placeholder)

                return self.W_placeholder[None, None, :].expand(
                    batch_size, 1, self.W_placeholder.size(-1)
                )
            else:

                return embeddings.gather(
                    1,
                    current_node[:, :, None].expand(batch_size, 1, embeddings.size(-1)),
                ).view(batch_size, 1, -1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(
            batch_size, num_steps, self.n_heads, 1, key_size
        ).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(
            glimpse_Q, glimpse_K.transpose(-2, -1)
        ) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[
                mask[None, :, :, None, :].expand_as(compatibility)
            ] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4)
            .contiguous()
            .view(-1, num_steps, 1, self.n_heads * val_size)
        )

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(
            -2
        ) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(
                v.size(0),
                v.size(1) if num_steps is None else num_steps,
                v.size(2),
                self.n_heads,
                -1,
            )
            .permute(
                3, 0, 1, 2, 4
            )  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
