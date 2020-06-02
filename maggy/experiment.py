#
#   Copyright 2020 Logical Clocks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

"""
Experiment module used for running asynchronous optimization tasks.

The programming model is that you wrap the code containing the model
training inside a wrapper function.
Inside that wrapper function provide all imports and parts that make up your
experiment, see examples below. Whenever a function to run an experiment is
invoked it is also registered in the Experiments service along with the
provided information.
"""
import atexit

from maggy.core import oblivious

lagom = None


def set_context(context, config):
    global lagom
    if context.lower() == "optimization":
        oblivious.searchspace = config
        lagom = oblivious.lagom_v1
    elif context.lower() == "ablation":
        lagom = oblivious.lagom_v1
    elif context.lower() == "dist_training":
        lagom = oblivious.mirrored
    else:
        raise Exception(
            "Distribution context {} is not supported. Has to be one of the "
            "following: `optimization`, `ablation` or `dist_training`.".format(context)
        )


atexit.register(oblivious._exit_handler)
