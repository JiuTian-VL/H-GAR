import sys
import os
import time
import click
import numpy as np
import torch
import dill
import hydra
import zmq


from h_gar.policy.base_image_policy import BaseImagePolicy
from h_gar.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_action,
)
from h_gar.common.pytorch_util import dict_apply
from h_gar.env_runner.robomimic_image_runner import undo_transform_action
import omegaconf
import traceback
import pickle
from omegaconf import open_dict

language_latents = pickle.load(open("prepared_data/language_latents.pkl", "rb"))
import torch
import torch.nn.functional as F


def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Extract unformatted traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # Print line of code where the exception occurred

    return "".join(tb_lines)

def smooth_action(act_out, window_size=3, pad_size=1):
    # Define the moving average kernel
    kernel = torch.ones(1, 1, window_size) / window_size  # 1x1x3 kernel
    kernel = kernel.to(act_out.device)  # Match device of the input tensor

    # Apply convolution with padding to preserve the sequence length
    # Unsqueeze the last dimension for convolution along the time axis
    print(act_out.shape)
    act_out_padded = F.pad(act_out, (0, 0, pad_size, pad_size), mode="replicate")

    batch_size, timesteps, action_dim = act_out_padded.shape
    act_out_padded = act_out_padded.permute(
        0, 2, 1
    )  # Shape: [batch_size, action_dim, timesteps]
    act_out_padded = act_out_padded.reshape(
        -1, 1, timesteps
    )  # Combine batch and action_dim

    smoothed_act_out = F.conv1d(act_out_padded, kernel, padding=0)

    smoothed_act_out = smoothed_act_out.reshape(
        batch_size, action_dim, timesteps - 2 * pad_size
    )
    smoothed_act_out = smoothed_act_out.permute(
        0, 2, 1
    )  # Shape: [batch_size, timesteps, action_dim]

    return smoothed_act_out

class PolicyInferenceNode:
    def __init__(
        self, ckpt_path: str, ip: str, port: int, device: str, output_dir: str
    ):
        self.ckpt_path = ckpt_path
        if not self.ckpt_path.endswith(".ckpt"):
            self.ckpt_path = os.path.join(self.ckpt_path, "checkpoints", "latest.ckpt")
        payload = torch.load(
            open(self.ckpt_path, "rb"), map_location="cpu", pickle_module=dill
        )
        self.cfg = payload["cfg"]

        
        with open_dict(self.cfg):
            if "autoregressive_model_params" in self.cfg.model.policy:
                self.cfg.model.policy.autoregressive_model_params.num_sampling_steps = (
                    "100"
                )
                print("-----------------------------------------------")
                print(
                    "num_sampling_steps",
                    self.cfg.model.policy.autoregressive_model_params.num_sampling_steps,
                )
                print("-----------------------------------------------")
        
        # export cfg to yaml
        cfg_path = self.ckpt_path.replace(".ckpt", ".yaml")
        with open(cfg_path, "w") as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
            print(f"Exported config to {cfg_path}")
        # print(f"Loading configure: {self.cfg.name}, workspace: {self.cfg._target_}, policy: {self.cfg.policy._target_}, model_name: {self.cfg.policy.obs_encoder.model_name}")
        print(
            f"Loading configure: {self.cfg.task.name}, workspace: {self.cfg.model._target_}, policy: {self.cfg.model.policy._target_}"
        )

        self.obs_res = get_real_obs_resolution(self.cfg.task.shape_meta)
        self.get_class_start_time = time.monotonic()

        cls = hydra.utils.get_class(self.cfg.model._target_)
        self.workspace = cls(self.cfg, output_dir=output_dir)
        self.workspace: BaseWorkspace
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy: BaseImagePolicy = self.workspace.model


        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
            print("Using EMA model")

        self.device = torch.device(device)
        self.policy.eval().to(self.device)
        self.policy.reset()
        self.ip = ip
        self.port = port

    def predict_action(self, obs_dict_np: dict, past_action_list=[]):

        if "task_name" in obs_dict_np:
            task_name = obs_dict_np["task_name"]
            print("task_name", obs_dict_np["task_name"])
            del obs_dict_np["task_name"]
        else:
            task_name = self.policy.task_name

        if self.cfg.task.dataset.language_emb_model is not None:
            # NOTE: codes changed here for other cases
            if "cup" in task_name:
                language_goal = language_latents["cup"]
                language_goal = torch.tensor(language_goal).to(self.device)
                language_goal = language_goal.unsqueeze(0)
            elif "towel" in task_name:
                language_goal = language_latents["towel"]
                language_goal = torch.tensor(language_goal).to(self.device)
                language_goal = language_goal.unsqueeze(0)
            elif "mouse" in task_name:
                language_goal = language_latents["mouse"]
                language_goal = torch.tensor(language_goal).to(self.device)
                language_goal = language_goal.unsqueeze(0)
            else:
                language_goal = None
            
            print("task_name", task_name)
        else:
            language_goal = None

        
        with torch.no_grad():
            obs_dict = dict_apply(
                obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device)
            )

            
            if self.cfg.name == "hgar":
                result = self.policy.predict_action(
                    obs_dict=obs_dict, language_goal=language_goal
                )

                past_action_list.append(np.array(result["action"][0].cpu()))
                if len(past_action_list) > 2:
                    past_action_list.pop(0)
                action = smooth_action(result["action_pred"].detach().to("cpu")).numpy()[0]
                

            else:
                result = self.policy.predict_action(
                    obs_dict, language_goal=language_goal
                )
                action = result["action_pred"][0].detach().to("cpu").numpy()
                print(f"action: {action}")

            

            del result
            del obs_dict

        return action, past_action_list

    def run_node(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.ip}:{self.port}")
        print(f"PolicyInferenceNode is listening on {self.ip}:{self.port}")

        past_action_list = []
        while True:
            obs_dict_np = socket.recv_pyobj()

            try:
                start_time = time.monotonic()
                action, past_action_list = self.predict_action(
                    obs_dict_np, past_action_list
                )
                print(f"Inference time: {time.monotonic() - start_time:.3f} s")

            except Exception as e:
                err_str = echo_exception()
                print(f"Error: {err_str}")
                action = err_str
            send_start_time = time.monotonic()
            # time.sleep(0.1)
            socket.send_pyobj(action)
            print(f"Send time: {time.monotonic() - send_start_time:.3f} s")


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--ip", default="0.0.0.0")
@click.option("--port", default=8766, help="Port to listen on")
@click.option("--device", default="cuda", help="Device to run on")
@click.option("--output_dir", required=True)
def main(input, ip, port, device, output_dir):

    node = PolicyInferenceNode(input, ip, port, device, output_dir)
    node.run_node()


if __name__ == "__main__":
    main()