import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import Union

# TODO: this runs but the values are unverified
# Note that "seed" can be chex.PRNGKey, this syntax is for compatibility with tfd


class UnitDiscreteActionDistribution:
    """Combine base and sap distributions along -1 axis"""
    def __init__(self, distribution_base_action: tfd.Categorical, distribution_sap_action: tfd.Categorical):
        self.distribution_base_action = distribution_base_action  # Categorical with N_BASE_ACTIONS
        self.distribution_sap_action = distribution_sap_action  # Categorical with N_SAP_ACTIONS

    def mode(self):  # -> shape: (..., 2)
        base_mode = self.distribution_base_action.mode()
        sap_mode = self.distribution_sap_action.mode()
        return jnp.stack((base_mode, sap_mode), axis=-1)

    def sample(self, seed) -> jnp.ndarray:  # -> shape: (..., 2)
        seed_base, seed_sap = jax.random.split(seed, 2)
        base_sample = self.distribution_base_action.sample(seed=seed_base)
        sap_sample = self.distribution_sap_action.sample(seed=seed_sap)
        return jnp.stack((base_sample, sap_sample), axis=-1)

    def log_prob(self, samples: jnp.ndarray) -> jnp.ndarray:
        # samples: (..., 2)
        base_sample, sap_sample = jnp.unstack(samples, axis=-1)  # (...), (...)
        log_prob_base = self.distribution_base_action.log_prob(base_sample)  # (...)
        log_prob_sap = self.distribution_sap_action.log_prob(sap_sample)  # (...)
        if_sap = log_prob_base + log_prob_sap
        if_notsap = log_prob_base
        cond = base_sample == len(self.distribution_base_action.logits) - 1  # if base action = sap, (...)
        return jnp.where(cond, if_sap, if_notsap)

    def sample_and_log_prob(self, seed) -> tuple[jnp.ndarray, jnp.ndarray]:
        samples = self.sample(seed)
        log_prob = self.log_prob(samples)
        return samples, log_prob

    def entropy(self):
        # H = H(base) + p(sap) * H(sap)
        H_base = self.distribution_base_action.entropy()
        H_sap = self.distribution_sap_action.entropy()
        p_sap = self.distribution_base_action.probs_parameter()[..., -1]
        return H_base + p_sap * H_sap


class AllUnitsActionDistribution:
    """Stack unit distributions along -1 axis"""
    def __init__(self, unit_distributions: Union[list[UnitDiscreteActionDistribution], list[tfd.Categorical]]):
        self.distributions = unit_distributions
        self.dim = len(self.distributions)

    def mode(self):
        modes = [distribution.mode() for distribution in self.distributions]
        return jnp.stack(modes, axis=-1)

    def sample(self, seed) -> jnp.ndarray:  # -> shape: (..., 2, N_MAX_UNITS)
        seeds = jax.random.split(seed, self.dim)
        samples = [distribution.sample(seed=seed) for (distribution, seed) in zip(self.distributions, seeds)]
        return jnp.stack(samples, axis=-1)

    def log_prob(self, samples: jnp.ndarray) -> jnp.ndarray:  # -> shape (...)
        # samples: (..., 2, N_MAX_UNITS)
        # log_prob (n-dimensional event) = sum (log_prob of event components)
        log_probs = [distribution.log_prob(sample) for (distribution, sample) in zip(self.distributions, jnp.unstack(samples, axis=-1))]
        return jnp.sum(jnp.stack(log_probs, axis=-1), axis=-1)

    def sample_and_log_prob(self, seed) -> tuple[jnp.ndarray, jnp.ndarray]:
        seeds = jax.random.split(seed, self.dim)
        samples = [distribution.sample(seed) for (distribution, seed) in zip(self.distributions, seeds)]
        log_probs = [distribution.log_prob(sample) for (distribution, sample) in zip(self.distributions, samples)]
        return jnp.stack(samples, axis=-1), jnp.sum(jnp.stack(log_probs, axis=-1), axis=-1)

    def entropy(self):
        # entropy of n-dimensional event = sum (entropy of independent event components)
        entropy = [distribution.entropy() for distribution in self.distributions]
        return jnp.sum(jnp.stack(entropy, axis=-1), axis=-1)


