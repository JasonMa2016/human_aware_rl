import torch
import numpy as np 

from torch.nn.utils.convert_parameters import (vector_to_parameters,
											   parameters_to_vector)
from torch.distributions.kl import kl_divergence

from human_aware_rl.maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from human_aware_rl.maml_rl.utils.optimization import conjugate_gradient

class MetaLearner(object):
    """
    A class object for policy-gradient based RL Meta Learner.
    """
    def __init__(self, sampler, policy, baseline, gamma=0.99,
                 fast_lr=0.05, tau=0.98, algorithm='MAML', device='cpu'):
        self.algorithm = algorithm
        self.sampler = sampler 
        self.policy = policy 
        self.baseline = baseline 
        self.gamma = gamma
        self.fast_lr = fast_lr 
        self.tau = tau 
        self.to(device)

    def inner_loss(self, episodes, params=None):
        """
        Compute the inner loss for the one-step gradient update. The inner
        loss is REINFORCE with baseline, computed on advantages estimated with
        Generalized Advantage Estimation.
        """
        # Compute baseline
        values = self.baseline(episodes)
        # Compute advantage
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)
        # Obtain policy
        pi = self.policy(episodes.observations, params=params)
        # Compute log probabilities
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        # Compute inner loss
        loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)

        return loss

    def adapt(self, episodes):
        """
        Adapt the parameters of the policy network to a new task, from
        sampled trajectories 'episodes', with a one-step gradient update. 
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        params, grads = self.policy.update_params(loss, step_size=self.fast_lr,
            algorithm=self.algorithm)

        return params, grads

    def sample(self, bc_agents):
        """
        Sample trajectories (before and after the update of the parameters)
        for all the tasks 'tasks'.
        """
        episodes = []
        grads = []
        successes = []
        for bc_model, bc_agent in bc_agents:
            # reset the current task 
            self.sampler.configure_bc_agent(bc_model, bc_agent)

            # collect meta-training set (current policy)
            train_episodes, train_successes = self.sampler.sample(self.policy, gamma=self.gamma,
                device=self.device)

            # compute adapted parameters
            params, task_grads = self.adapt(train_episodes)

            # collect meta-testing set (adapted policy)
            test_episodes, test_successes = self.sampler.sample(self.policy, params=params,
                gamma=self.gamma, device=self.device)

            episodes.append((train_episodes, test_episodes))
            successes.append(test_successes)
            grads.append(task_grads)
        return episodes, grads, np.array(successes)

    def kl_divergence(self, episodes, old_pis=None):
        """
        Computes the KL-divergence between the original and the adapted policy.
        """
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, test_episodes), old_pi in zip(episodes, old_pis):
            params, task_grads = self.adapt(train_episodes)
            pi = self.policy(test_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = test_episodes.mask
            if test_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))


    def hessian_vector_product(self, episodes, damping=1e-2):
        """
        Hessian-vector product, based on the Perlmutter method.
        """
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        """
        TRPO surrogate loss.
        """
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, test_episodes), old_pi in zip(episodes, old_pis):
            # Compute the adapted parameters
            params, task_grads = self.adapt(train_episodes)
            # Compute the outer (TRPO) loss using the adapted parameters
            with torch.set_grad_enabled(old_pi is None):

                # Compute the adapted policy
                pi = self.policy(test_episodes.observations, params=params) # normal policy
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(test_episodes)
                advantages = test_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=test_episodes.mask)

                log_ratio = (pi.log_prob(test_episodes.actions)
                    - old_pi.log_prob(test_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                # Lower bound of the actual loss; see the TRPO paper for a detailed derivation
                loss = -weighted_mean(ratio * advantages, dim=0,
                    weights=test_episodes.mask)
                losses.append(loss)

                # Compute the kl divergence between the new and the old policies
                mask = test_episodes.mask
                if test_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0,
                    weights=mask)
                kls.append(kl)


        return  (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def step(self, episodes, max_kl=1e-2, cg_iters=10, cg_damping=1e-2,
        ls_max_steps=10, ls_backtrack_ratio=0.5):
        """
        Meta-optimization step (ie. update of the initial parameters), based on
        TRPO.
        """

        # Compute outer loss and adapted policies
        old_loss, _, old_pis = self.surrogate_loss(episodes)

        # Compute gradient of the outer loss with respect to the original parameters
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
            cg_iters=cg_iters)

        # Compute the natural policy gradient
        natural_gradient = torch.sqrt(2*max_kl / torch.dot(stepdir, hessian_vector_product(stepdir))) * stepdir

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search for TRPO
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * natural_gradient,
                self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss 
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            # Keep decreasing the step-size until the trust region constraint is satisfied
            step_size *= ls_backtrack_ratio 
        else:
            vector_to_parameters(old_params, self.policy.parameters())

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device


