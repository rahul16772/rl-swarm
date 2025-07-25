import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from genrl.blockchain import SwarmCoordinator
from genrl.communication import Communication
from genrl.communication.hivemind.hivemind_backend import HivemindBackend
from genrl.data import DataManager
from genrl.game import BaseGameManager
from genrl.game.game_manager import DefaultGameManagerMixin
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.system_utils import get_system_info
from genrl.rewards import RewardManager
from genrl.roles import RoleManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from huggingface_hub import login, whoami

from rgym_exp.src.utils.name_utils import get_name_from_peer_id


class SwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """GameManager that orchestrates a game using a SwarmCoordinator."""

    def __init__(
        self,
        coordinator: SwarmCoordinator,
        max_stage: int,
        max_round: int,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        log_dir: str = "logs",
        hf_token: str | None = None,
        hf_push_frequency: int = 20,
        **kwargs,
    ):
        """
        Initializes the SwarmGameManager.

        This constructor sets up the game environment, including logging, communication,
        blockchain coordination, and integration with the Hugging Face Hub. It also
        handles specific configurations for running with or without vLLM.
        """
        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode,
        )

        assert isinstance(self.communication, HivemindBackend)
        self.train_timeout = 60 * 60 * 24 * 31  # 1 month

        # Logging Setup
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)
        format_msg = f"[{self.animal_name}] %(asctime)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format_msg)
        formatter = logging.Formatter(format_msg)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{self.animal_name}.log")
        )
        file_handler.setFormatter(formatter)
        _LOG = get_logger()
        _LOG.addHandler(file_handler)

        # Register peer_id and get current round from the chain
        self.coordinator = coordinator
        self.coordinator.register_peer(self.peer_id)
        round, _ = self.coordinator.get_round_and_stage()
        self.state.round = round
        self.communication.step_ = (
            self.state.round
        )  # initialize communication module to contract's round

        get_logger().info(
            f"🐱 Hello 🐈 [{get_name_from_peer_id(self.peer_id)}] 🦮 [{self.peer_id}]!"
        )
        get_logger().info(f"bootnodes: {kwargs.get('bootnodes', [])}")
        
        # --- VLLM INTEGRATION START ---
        # Safely get the model name first, then use it.
        model_name = "UnknownModel"
        
        # Check if we are in vLLM mode
        if hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm:
            # In vLLM mode, use the name we saved in the trainer
            model_name = getattr(self.trainer, "model_name", "vLLM_Model")
        else:
            # In standard training mode, safely access the config attribute
            config_obj = getattr(getattr(self.trainer, "model", None), "config", None)
            if config_obj:
                model_name = getattr(config_obj, "_name_or_path", "UnknownModel")
        
        get_logger().info(f"Using Model: {model_name}")

        # Enable push to HF if token was provided
        self.hf_token = hf_token
        if self.hf_token not in [None, "None"]:
            # This block should only run if we can actually push, which means we're NOT in vLLM mode.
            if not (hasattr(self.trainer, "use_vllm") and self.trainer.use_vllm):
                try:
                    username = whoami(token=self.hf_token)["name"]
                    model_name_suffix = model_name.split("/")[-1]
                    hub_model_id = f"{username}/{model_name_suffix}-Gensyn-Swarm-{self.animal_name}"
                    
                    self.trainer.args.hub_model_id = hub_model_id
                    self.trainer.args.push_to_hub = True
                    self.trainer.args.hub_token = self.hf_token
                    self.hf_push_frequency = hf_push_frequency
                    get_logger().info("Logging into Hugging Face Hub...")
                    login(self.hf_token)
                except Exception as e:
                    get_logger().warning(f"Could not set up Hugging Face push. Error: {e}")
            else:
                get_logger().info("Hugging Face push is disabled in vLLM mode.")
        # --- VLLM INTEGRATION END ---

        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

        # Time-based submission attributes
        self.batched_signals = 0.0
        self.time_since_submit = time.time() #seconds
        self.submit_period = 3.0 #hours
        self.submitted_this_round = False

    def _get_total_rewards_by_agent(self):
        """Aggregates total rewards for each agent across all stages."""
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            rewards = self.rewards[stage]
            for agent_id, agent_rewards in rewards.items():
                for batch_id, batch_rewards in agent_rewards.items():
                    tot = 0
                    for generation_rewards in batch_rewards:
                        tot += sum(generation_rewards)
                    rewards_by_agent[agent_id] += tot
        return rewards_by_agent

    def _get_my_rewards(self, signal_by_agent):
        """Calculates the adjusted reward signal for the current agent."""
        my_signal = signal_by_agent.get(self.peer_id, 0)
        my_signal = (my_signal + 1) * (my_signal > 0) + my_signal * (
            my_signal <= 0
        )
        return my_signal

    def _try_submit_to_chain(self, signal_by_agent):
        """Submits rewards and winners to the chain if the submission period has elapsed."""
        elapsed_time_hours = (time.time() - self.time_since_submit) / 3600
        if elapsed_time_hours > self.submit_period:
            get_logger().info(f"Submitting batched signal of {int(self.batched_signals)} to the chain.")
            self.coordinator.submit_reward(
                self.state.round, 0, int(self.batched_signals), self.peer_id
            )
            self.batched_signals = 0.0
            
            if signal_by_agent:
                max_agent, max_signal = max(signal_by_agent.items(), key=lambda x: x[1])
                self.coordinator.submit_winners(self.state.round, [max_agent], self.peer_id)
            
            self.time_since_submit = time.time()
            self.submitted_this_round = True

    def _hook_after_rewards_updated(self):
        """Hook called after rewards are updated. Batches signals and tries to submit."""
        signal_by_agent = self._get_total_rewards_by_agent()
        self.batched_signals += self._get_my_rewards(signal_by_agent)
        self._try_submit_to_chain(signal_by_agent)

    def _hook_after_round_advanced(self):
        """Hook called after the round advances."""
        self._save_to_hf()

        # Try to submit to chain again if necessary, but don't update our signal twice
        if not self.submitted_this_round:
            signal_by_agent = self._get_total_rewards_by_agent()
            self._try_submit_to_chain(signal_by_agent)
        
        # Reset flag for next round
        self.submitted_this_round = False

        # Block until swarm round advances
        self.agent_block()

    def _hook_after_game(self):
        """Hook called after the game finishes. Performs a final save to the HF hub."""
        self._save_to_hf()

    def _save_to_hf(self):
        """Saves the model to the Hugging Face Hub if configured."""
        if (
            self.hf_token not in [None, "None"]
            and self.state.round % self.hf_push_frequency == 0
        ):
            get_logger().info(f"Pushing model to huggingface for round {self.state.round}")
            try:
                repo_id = self.trainer.args.hub_model_id
                if repo_id is None:
                    repo_id = Path(self.trainer.args.output_dir).name

                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=[
                        "rl-swarm",
                        "genrl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {self.animal_name}",
                    ],
                )
            except Exception:
                get_logger().exception(
                    "Failed to push model to the Hugging Face Hub. When you conclude training please try manually pushing it yourself using the instructions here: https://huggingface.co/docs/hub/en/models-uploading",
                    stack_info=True,
                )

    def agent_block(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15
    ):
        """Blocks execution until the coordinator signals the next round."""
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = (
            check_interval  # Exponential backoff for already finished rounds.
        )
        
        get_logger().info(f"Waiting for round {self.state.round} to complete...")

        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            _ = self.communication.dht.get_visible_maddrs(latest=True)

            # Retrieve current round and stage.
            try:
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(
                        f"Could not fetch round and stage: {e}. Next check in {check_interval}s."
                    )
                    fetch_log_time = curr_time

                time.sleep(check_interval)
                continue

            if round_num >= self.state.round:
                get_logger().info(f"🐝 Joining round: {round_num}")
                check_backoff = check_interval  # Reset backoff after successful round
                self.state.round = round_num  # advance to swarm's round.
                return
            else:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().info(
                        f"Still waiting for round {self.state.round}. Current swarm round is {round_num}. Next check in {check_backoff:.1f}s."
                    )
                    fetch_log_time = curr_time
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.max_round - 1:
                get_logger().info("Max round reached. Concluding training.")
                return

        get_logger().warning("Training timed out!")
