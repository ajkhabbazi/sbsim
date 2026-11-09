"""Step 7b/7c: train the same SAC agent on EnergyPlus or SBSim and time it.

Follows SBSim's own training script (smart_control/reinforcement_learning/
scripts/train.py): SBSim's SAC agent with its default networks, replay buffer
of 50,000 frames sampled in pairs (num_steps=2), batch 256, and loops of
"collect steps, then gradient updates". Differences from train.py:
  - a TF-Agents uniform replay buffer replaces Reverb (Reverb has no Windows build)
  - no starter buffer, evaluation, metrics or summaries; each loop starts with
    collection, so the buffer has data before the first update
  - the time spent inside the simulator is measured separately

  - the actor network is rebuilt here with the wrapper from SBSim's SAC_Demo
    notebook, because SBSim's sac_networks.py raises a TypeError (see below)

Episodes: EnergyPlus runs one year per episode, SBSim one day (sim_config_1_day.gin).

Usage (in the sbsim-bench env):
  python train_bench.py --sim eplus
  python train_bench.py --sim sbsim --iterations 1 --collect-steps 12
"""
import os
# as in SBSim's train.py: must be set before TensorFlow is imported
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"

import argparse, datetime, json, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import wrappers
from tf_agents.networks import nest_map, sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.train import actor, learner
from tf_agents.train.utils import spec_utils
from tf_agents.utils import nest_utils

from bench_paths import SBSIM_DIR
SBSIM = str(SBSIM_DIR)
sys.path.insert(0, SBSIM)
from smart_control.reinforcement_learning.agents.sac_agent import create_sac_agent
from smart_control.reinforcement_learning.agents.networks.sac_networks import dense

HERE = Path(__file__).resolve().parent
REPLAY_CAPACITY = 50000      # train.py's replay buffer capacity
ACTOR_FC_LAYERS = (256, 256)  # create_sac_agent's default


class _TanhNormalProjectionNetworkWrapper(
        tanh_normal_projection_network.TanhNormalProjectionNetwork):
    """SBSim's wrapper (sac_networks.py:106) with the notebook's call signature.

    sac_networks.py:115 drops `network_state=()`, which NestMap passes in
    tf-agents 0.18, so SBSim's own actor network raises
    "call() got an unexpected keyword argument 'network_state'".
    SAC_Demo.ipynb keeps the argument, as here.
    """

    def __init__(self, sample_spec, predefined_outer_rank=1):
        super().__init__(sample_spec)
        self.predefined_outer_rank = predefined_outer_rank

    def call(self, inputs, network_state=(), **kwargs):
        kwargs["outer_rank"] = self.predefined_outer_rank
        if "step_type" in kwargs:
            del kwargs["step_type"]
        return super().call(inputs, **kwargs)


def create_actor_network(actor_fc_layers, action_tensor_spec):
    """Same layers as sac_networks.create_sequential_actor_network."""
    def tile_as_nest(non_nested_output):
        return tf.nest.map_structure(lambda _: non_nested_output, action_tensor_spec)

    return sequential.Sequential(
        [dense(num_units) for num_units in actor_fc_layers]
        + [tf.keras.layers.Lambda(tile_as_nest)]
        + [nest_map.NestMap(tf.nest.map_structure(
            _TanhNormalProjectionNetworkWrapper, action_tensor_spec))])


class TimedEnv(wrappers.PyEnvironmentBaseWrapper):
    """Counts the time spent inside the simulator's reset() and step()."""

    def __init__(self, env):
        super().__init__(env)
        self.step_seconds = 0.0
        self.reset_seconds = 0.0
        self.steps = 0
        self.resets = 0

    @property
    def sim_seconds(self):
        return self.step_seconds + self.reset_seconds

    def _reset(self):
        t0 = time.perf_counter()
        time_step = self._env.reset()
        self.reset_seconds += time.perf_counter() - t0
        self.resets += 1
        return time_step

    def _step(self, action):
        t0 = time.perf_counter()
        time_step = self._env.step(action)
        self.step_seconds += time.perf_counter() - t0
        self.steps += 1
        return time_step


def make_env(sim, run_dir):
    if sim == "eplus":
        from eplus_tf_env import EnergyPlusTFEnv
        return EnergyPlusTFEnv(out_dir=str(run_dir / "eplus"))
    from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
    from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment
    return create_and_setup_environment(
        os.path.join(CONFIG_PATH, "sim_config_1_day.gin"), metrics_path=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", choices=["eplus", "sbsim"], required=True)
    p.add_argument("--iterations", type=int, default=6, help="collect-then-train loops")
    p.add_argument("--collect-steps", type=int, default=48, help="simulator steps per loop (48 = 4 hours)")
    p.add_argument("--learner-iterations", type=int, default=200, help="gradient updates per loop (train.py default)")
    p.add_argument("--batch-size", type=int, default=256, help="train.py default")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    stamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    run_dir = HERE / "out-train" / f"{args.sim}-{stamp}"   # fresh, so nothing is restored
    run_dir.mkdir(parents=True)

    # ---- setup (not part of the timed loop) ------------------------------
    t_setup = time.perf_counter()
    env = TimedEnv(make_env(args.sim, run_dir))
    _, action_spec, time_step_spec = spec_utils.get_tensor_specs(env)
    train_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    agent = create_sac_agent(
        time_step_spec=time_step_spec, action_spec=action_spec,
        actor_network=create_actor_network(ACTOR_FC_LAYERS, action_spec),
        train_step_counter=train_step)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec, batch_size=1, max_length=REPLAY_CAPACITY)

    def add_to_buffer(trajectory):
        replay_buffer.add_batch(nest_utils.batch_nested_array(trajectory))

    collect_actor = actor.Actor(   # resets the simulator once
        env, py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy), train_step,
        steps_per_run=args.collect_steps, observers=[add_to_buffer])
    setup_seconds = time.perf_counter() - t_setup
    first_reset_seconds = env.reset_seconds

    # ---- timed loop ------------------------------------------------------
    agent_learner = None
    rows = []
    print(f"{'loop':>4} {'sim steps':>9} {'simulator s':>11} {'collect s':>9} {'train s':>8}")
    for i in range(args.iterations):
        sim_before = env.sim_seconds
        t0 = time.perf_counter()
        collect_actor.run()
        collect_s = time.perf_counter() - t0
        sim_s = env.sim_seconds - sim_before

        if agent_learner is None:   # built once the buffer has data to sample
            dataset = replay_buffer.as_dataset(
                sample_batch_size=args.batch_size, num_steps=2, num_parallel_calls=3).prefetch(3)
            t1 = time.perf_counter()
            agent_learner = learner.Learner(
                root_dir=str(run_dir / "learner"), train_step=train_step, agent=agent,
                experience_dataset_fn=lambda: dataset, summary_interval=0, checkpoint_interval=0)
            learner_setup_seconds = time.perf_counter() - t1

        t0 = time.perf_counter()
        loss = agent_learner.run(iterations=args.learner_iterations)
        train_s = time.perf_counter() - t0
        loss_value = float(loss.loss)
        if not np.isfinite(loss_value):
            raise RuntimeError(f"loss is {loss_value} at loop {i}")

        rows.append(dict(loop=i, sim_steps=env.steps, simulator_s=sim_s,
                         collect_s=collect_s, train_s=train_s, loss=loss_value))
        print(f"{i:>4} {env.steps:>9} {sim_s:>11.2f} {collect_s:>9.2f} {train_s:>8.2f}")

    env.close()

    # ---- summary ---------------------------------------------------------
    sim_s = sum(r["simulator_s"] for r in rows)
    collect_s = sum(r["collect_s"] for r in rows)
    train_s = sum(r["train_s"] for r in rows)
    summary = dict(
        sim=args.sim, args=vars(args), run_dir=str(run_dir),
        sim_steps=env.steps, resets=env.resets,
        setup_s=setup_seconds, first_reset_s=first_reset_seconds,
        learner_setup_s=learner_setup_seconds,
        simulator_s=sim_s, simulator_ms_per_step=1000 * env.step_seconds / env.steps,
        collect_s=collect_s, agent_and_buffer_s=collect_s - sim_s,
        train_s=train_s, train_s_after_first_loop=sum(r["train_s"] for r in rows[1:]),
        gradient_updates=int(train_step.numpy()), final_loss=rows[-1]["loss"],
        loops=rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"simulator steps: {env.steps}   resets: {env.resets}   gradient updates: {summary['gradient_updates']}")
    print(f"setup: {setup_seconds:.2f} s (includes first reset {first_reset_seconds:.2f} s)"
          f"   learner setup: {learner_setup_seconds:.2f} s")
    print(f"simulator: {sim_s:.2f} s   ({summary['simulator_ms_per_step']:.3f} ms per step)")
    print(f"agent + replay buffer during collection: {summary['agent_and_buffer_s']:.2f} s")
    print(f"training: {train_s:.2f} s   ({summary['train_s_after_first_loop']:.2f} s after the first loop)")
    print(f"final loss: {summary['final_loss']:.4g}")
    print(f"saved: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
