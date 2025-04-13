# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from llamafactory.train.tuner import run_exp  # use absolute import


def launch():
    #### for debug #########
    # print(f"Parent PID (PPID): {os.getppid()}")
    # print(f"Current PID: {os.getpid()}")
    # print(f"UID: {os.getuid()}, EUID: {os.geteuid()}") 
    # input("Press Enter to run the experiment...\n")
    run_exp()


if __name__ == "__main__":
    launch()
