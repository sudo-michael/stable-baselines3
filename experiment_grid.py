# Modified version of ExperimentGrid from Omnisafe
# src: https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/common/experiment_grid.py
# Changes:
# -
# -
# -
# copyright 2023 omnisafe team. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.
# ==============================================================================

import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor as Pool
from copy import deepcopy
from typing import Any

from rich.console import Console

import hashlib


class ExperimentGrid:
    def __init__(self, experiment_name: str, config: dict[str, Any]) -> None:
        self.experiment_name = experiment_name
        self.config = config
        self.log_dir = os.path.join("./", "runs", self.experiment_name)
        self._console: Console = Console()

    def print_config(self):
        self._console.print(
            f"Grid Search \[{self.experiment_name}] over the following parameters",
            style="green bold",
        )
        total_algo_variants = 0
        for key, value in self.config.items():
            self._console.print("", key.ljust(40), style="cyan bold")

            if key == "algo":
                for algo_name, algo_config in value.items():
                    self._console.print("[white bold]\t" + json.dumps(algo_name, indent=4, sort_keys=True))
                    algo_variants = 1
                    for k, v in algo_config.items():
                        self._console.print(f"\t [yellow bold]{k}[/yellow bold]: {v}")
                        algo_variants *= len(v)
                    total_algo_variants += algo_variants
            else:
                for _, val in enumerate(value):
                    self._console.print("[white bold]\t" + json.dumps(val, indent=4, sort_keys=True))

        # Count up the number of variants. The number counting seeds
        # is the total number of experiments that will run; the number not
        # counting seeds is the total number of otherwise-unique configs
        # being investigated.
        nvars_seedless = len(self.config["env_id"]) * total_algo_variants
        self._console.print(" Variants, not counting seeds: ".ljust(40), nvars_seedless, style="green bold")

    def save_grid_config(self) -> None:
        """Save experiment grid configurations as json."""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "grid_config.json")
        self._console.print(
            "Save with config of experiment grid in grid_config.json",
            style="yellow bold",
        )
        json_config = json.dumps(self.config, indent=4)
        with open(path, encoding="utf-8", mode="w") as f:
            f.write(json_config)

    def _variants(self, keys: list[str], vals: list[Any]) -> list[dict[str, Any]]:
        """Recursively builds list of valid variants.

        Args:
            keys (keys: list[str]): List of keys.
            vals (list[Any]): List of values.

        Returns:
            List of valid variants.
        """
        if len(keys) == 1:
            pre_variants: list[dict[str, Any]] = [{}]
        else:
            pre_variants = self._variants(keys[1:], vals[1:])

        variants = []
        for val in vals[0]:
            for pre_v in pre_variants:
                current_variants = deepcopy(pre_v)
                v_temp = {}
                key_list = keys[0].split(":")
                v_temp[key_list[-1]] = val
                for key in reversed(key_list[:-1]):
                    v_temp = {key: v_temp}
                self.update_dict(current_variants, v_temp)
                variants.append(current_variants)

        return variants

    def variants(self):
        flat_variants = []
        for algo in self.config["algo"]:
            keys = ["env_id", "algo", *self.config["algo"][algo].keys(), "seed"]
            values = [self.config["env_id"], [algo], *self.config["algo"][algo].values(), self.config["seed"]]
            flat_variants.extend(self._variants(keys, values))

        def check_duplicate(var: dict[str, Any]) -> dict[str, Any]:
            """Build the full nested dict version of var, based on key names."""
            new_var: dict[str, Any] = {}
            unflatten_set: set = set()

            for key, value in var.items():
                assert key not in new_var, "You can't assign multiple values to the same key."
                new_var[key] = value

            # make sure to fill out the nested dict.
            for key in unflatten_set:
                new_var[key] = check_duplicate(new_var[key])

            return new_var

        return [check_duplicate(var) for var in flat_variants]

    def variants_for_env_id_algo(self, env_id, algo):
        assert env_id in self.config['env_id']
        assert algo in self.config['algo']
        keys = ["env_id", "algo", *self.config["algo"][algo].keys()]
        values = [[env_id], [algo], *self.config["algo"][algo].values()]
        flat_variants = self._variants(keys, values)

        def check_duplicate(var: dict[str, Any]) -> dict[str, Any]:
            """Build the full nested dict version of var, based on key names."""
            new_var: dict[str, Any] = {}
            unflatten_set: set = set()

            for key, value in var.items():
                assert key not in new_var, "You can't assign multiple values to the same key."
                new_var[key] = value

            # make sure to fill out the nested dict.
            for key in unflatten_set:
                new_var[key] = check_duplicate(new_var[key])

            return new_var

        return [check_duplicate(var) for var in flat_variants]

    def run(self, thunk, num_pool=1):
        try:
            assert not os.path.exists(self.log_dir)
        except AssertionError:
            self._console.print(f"ERROR: log_dir {self.log_dir} already exists!", style="bold red")

        self.save_grid_config()
        self.print_config()

        # make the list of all variants.
        variants = self.variants()

        results = []  # holds True / False value if done
        exp_names = []

        with Pool(max_workers=num_pool, mp_context=mp.get_context("spawn")) as executor:
            for idx, var in enumerate(variants):
                exp_name = json.dumps(variants, sort_keys=True)
                hashed_exp_name = self.hash_varient(var)
                exp_names.append(f"{hashed_exp_name[:5]}:{exp_name}")

                # save entire variant config to file in log_dir
                exp_log_dir = os.path.join(self.log_dir, hashed_exp_name)
                os.makedirs(exp_log_dir, exist_ok=True)
                path = os.path.join(exp_log_dir, "exps_config.json")
                json_config = json.dumps(var, indent=4)
                with open(path, encoding="utf-8", mode="a+") as f:
                    f.write("\n" + json_config)

                # pop all non-algorithm specific parameters from the variant
                algo = var.pop("algo")
                env_id = var.pop("env_id")
                seed = var.pop("seed", 0)

                results.append(
                    executor.submit(
                        thunk,
                        experiment_id=idx,
                        exp_log_dir=exp_log_dir,
                        algo=algo,
                        env_id=env_id,
                        seed=seed,
                        algo_params=var,
                    )
                )

        # save results
        path = os.path.join(self.log_dir, "results.txt")
        str_len = max(len(exp_name) for exp_name in exp_names)
        exp_names = [exp_name.ljust(str_len) for exp_name in exp_names]
        with open(path, "a+", encoding="utf-8") as f:
            for idx, _ in enumerate(variants):
                if is_experiment_done := results[idx].result():
                    f.write(exp_names[idx] + ": ")
                    f.write("finished:" + str(is_experiment_done) + ",")
                    f.write("\n")

    def update_dict(self, total_dict: dict[str, Any], item_dict: dict[str, Any]) -> None:
        """Updater of multi-level dictionary.

        Args:
            total_dict (dict[str, Any]): The total dictionary.
            item_dict (dict[str, Any]): The item dictionary.

        Examples:
            >>> total_dict = {'a': {'b': 1, 'c': 2}}
            >>> item_dict = {'a': {'b': 3, 'd': 4}}
            >>> update_dict(total_dict, item_dict)
            >>> total_dict
            {'a': {'b': 3, 'c': 2, 'd': 4}}
        """
        for idd in item_dict:
            total_value = total_dict.get(idd)
            item_value = item_dict.get(idd)

            if total_value is None:
                total_dict.update({idd: item_value})
            elif isinstance(item_value, dict):
                self.update_dict(total_value, item_value)
                total_dict.update({idd: total_value})
            else:
                total_value = item_value
                total_dict.update({idd: total_value})

    def hash_varient(self, varient: dict[str, Any]) -> str:
        """Generate the folder name for a varient (excluding the seed). """
        name_varient = deepcopy(varient)
        name_varient.pop("seed", 0)
        exp_name = json.dumps(name_varient, sort_keys=True)
        hashed_exp_name = name_varient["env_id"][:30] + "---" + self.hash_string(exp_name)
        return hashed_exp_name

    def hash_string(self, string: str) -> str:
        salt = b"\xf8\x99/\xe4\xe6J\xd8d\x1a\x9b\x8b\x98\xa2\x1d\xff3*^\\\xb1\xc1:e\x11M=PW\x03\xa5\\h"
        # convert string to bytes and add salt
        salted_string = salt + string.encode("utf-8")
        # use sha256 to hash
        hash_object = hashlib.sha256(salted_string)
        # get the hex digest
        return hash_object.hexdigest()
