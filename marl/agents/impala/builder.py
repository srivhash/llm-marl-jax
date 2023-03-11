"""IMPALA Builder."""

from typing import Any, Callable, Iterator, Optional

import haiku as hk
import optax
import reverb
import rlax
from acme import core
from acme.jax import networks as networks_lib
from acme.utils import counting, loggers

from marl import specs as ma_specs
from marl import types
from marl.agents.builder import MABuilder
from marl.agents.impala.config import IMPALAConfig
from marl.agents.impala.learning import IMPALALearner, PopArtIMPALALearner
from marl.modules import popart_simple


class IMPALABuilder(MABuilder):
  """MAIMPALA Builder."""

  def __init__(
      self,
      config: IMPALAConfig,
      core_state_spec: hk.LSTMState,
      table_extension: Optional[Callable[[], Any]] = None,
  ):
    super().__init__(config, core_state_spec, table_extension)

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: types.RecurrentNetworks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: ma_specs.MAEnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec, replay_client

    optimizer = optax.chain(
        optax.clip_by_global_norm(self._config.max_gradient_norm),
        optax.rmsprop(
            self._config.learning_rate,
            decay=self._config.rmsprop_decay,
            eps=self._config.rmsprop_eps,
            initial_scale=self._config.rmsprop_init,
            momentum=self._config.rmsprop_momentum
            if self._config.rmsprop_momentum != 0 else None,
        ),
    )

    return IMPALALearner(
        network=networks,
        iterator=dataset,
        optimizer=optimizer,
        n_agents=self._config.n_agents,
        random_key=random_key,
        discount=self._config.discount,
        entropy_cost=self._config.entropy_cost,
        baseline_cost=self._config.baseline_cost,
        max_abs_reward=self._config.max_abs_reward,
        counter=counter,
        logger=logger_fn(label="learner"),
    )


class PopArtIMPALABuilder(MABuilder):
  """MAIMPALA Builder."""

  def __init__(
      self,
      config: IMPALAConfig,
      core_state_spec: hk.LSTMState,
      table_extension: Optional[Callable[[], Any]] = None,
  ):
    super().__init__(config, core_state_spec, table_extension)

  def make_learner(
      self,
      random_key: networks_lib.PRNGKey,
      networks: types.RecurrentNetworks,
      dataset: Iterator[reverb.ReplaySample],
      logger_fn: loggers.LoggerFactory,
      environment_spec: ma_specs.MAEnvironmentSpec,
      replay_client: Optional[reverb.Client] = None,
      counter: Optional[counting.Counter] = None,
  ) -> core.Learner:
    del environment_spec, replay_client

    optimizer = optax.chain(
        optax.clip_by_global_norm(self._config.max_gradient_norm),
        optax.rmsprop(
            self._config.learning_rate,
            decay=self._config.rmsprop_decay,
            eps=self._config.rmsprop_eps,
            initial_scale=self._config.rmsprop_init,
            momentum=self._config.rmsprop_momentum
            if self._config.rmsprop_momentum != 0 else None,
        ),
    )

    def _popart(axis):
      init_fn, update_fn = rlax.popart(
          num_outputs=1,
          step_size=self._config.step_size,
          scale_lb=self._config.scale_lb,
          scale_ub=self._config.scale_ub,
          axis_name=axis)
      return types.PopArtLayer(init_fn=init_fn, update_fn=update_fn)

    def _art(axis):
      init_fn, update_fn = popart_simple(
          num_outputs=1,
          step_size=self._config.step_size,
          scale_lb=self._config.scale_lb,
          scale_ub=self._config.scale_ub,
          axis_name=axis)
      return types.PopArtLayer(init_fn=init_fn, update_fn=update_fn)

    return PopArtIMPALALearner(
        network=networks,
        popart=(_art if self._config.only_art else _popart,
                self._config.only_art),
        iterator=dataset,
        optimizer=optimizer,
        n_agents=self._config.n_agents,
        random_key=random_key,
        discount=self._config.discount,
        entropy_cost=self._config.entropy_cost,
        baseline_cost=self._config.baseline_cost,
        max_abs_reward=self._config.max_abs_reward,
        counter=counter,
        logger=logger_fn(label="learner"),
    )