# Copyright 2025 nCompass Technologies
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

"""Immutable base class and mutate decorator for controlled state management."""


def mutate(func):
    """Decorator to make a method mutable by temporarily allowing attribute changes."""
    def wrapper(self, *args, **kwargs):
        # Store current state of attributes
        stored_attrs = self.attrWasSet.copy()
        # Clear attribute list to allow changes
        self.attrWasSet.clear()
        
        try:
            # Execute the function
            result = func(self, *args, **kwargs)
            # Restore attribute protection
            self.attrWasSet = stored_attrs
            return result
        except Exception as e:
            # Restore attribute protection even if function fails
            self.attrWasSet = stored_attrs
            raise e
            
    return wrapper


class Immutable:
    """Base class that prevents attribute modification after first set.
    
    Once an attribute is set on an Immutable instance, it cannot be changed.
    Use the @mutate decorator on methods that need to modify attributes.
    """
    
    def __new__(cls):
        instance = super().__new__(cls)
        instance.attrWasSet = []
        return instance

    def __setattr__(self, name, value):
        if name == 'attrWasSet':
            super().__setattr__(name, value)
        elif name in self.attrWasSet:
            raise RuntimeError(f'Cannot change state of {name} once created')
        else:
            self.attrWasSet.append(name)
            super().__setattr__(name, value)

