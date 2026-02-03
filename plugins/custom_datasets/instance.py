"""
Instance class
===============
A generic instance class for storing and manipulating data with flexible attributes.
"""
import copy
from typing import List, Any, Dict, Optional
from conversation import Conversation


class Instance:
    def __init__(self, *, 
                 text: str = None,
                 prompt: str = None,
                 responses: List[str] = None,
                 labels: List[Any] = None,
                 metadata: Dict = None,
                 parents: List['Instance'] = None,
                 children: List['Instance'] = None,
                 **kwargs):
        """
        Initializes a generic instance with various attributes.

        :param str text: The primary text content associated with the instance.
        :param str prompt: A prompt or template string associated with the instance.
        :param List[str] responses: A list of responses or outputs.
        :param List[Any] labels: A list of labels or annotations.
        :param Dict metadata: A dictionary for storing additional metadata.
        :param List[Instance] parents: A list of parent instances, indicating lineage or history.
        :param List[Instance] children: A list of child instances, indicating derived instances.
        :param kwargs: Additional keyword arguments.
        """
        self._data = {}
        self.text = text
        self.prompt = prompt
        
        if responses is None:
            responses = []
        self.responses = responses
        
        if labels is None:
            labels = []
        self.labels = labels
        
        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata
            
        if parents is None:
            parents = []
        self.parents = parents
        
        if children is None:
            children = []
        self.children = children
        
        self.index: Optional[int] = None
        self._data.update(**kwargs)

    def copy(self):
        """
        Creates a deep copy of the instance.

        :return Instance: A new instance that is a deep copy of the current instance.
        """
        new_instance = Instance(
            text=self.text,
            prompt=self.prompt,
            responses=self.responses.copy(),
            labels=self.labels.copy(),
            metadata=copy.deepcopy(self.metadata),
            parents=[i for i in self.parents],
            children=[i for i in self.children]
        )
        rest_data = {key: value for key, value in self._data.items() 
                     if key not in new_instance._data}
        new_instance._data.update(copy.deepcopy(rest_data))
        return new_instance
    
    def shallow_copy(self):
        """
        Creates a shallow copy of the instance.

        :return Instance: A new instance that is a shallow copy of the current instance.
        """
        new_instance = Instance(
            text=self.text,
            prompt=self.prompt,
            responses=self.responses.copy(),
            labels=self.labels.copy()
        )
        return new_instance

    def delete(self, *keys):
        """
        Deletes specified attributes from the instance.

        :param keys: The keys of the attributes to be deleted.
        """
        for key in keys:
            if key in self._data:
                del self._data[key]

    def to_conversation(self):
        """
        Converts the instance into a Conversation object.

        :return Conversation: A Conversation representation of the instance.
        """
        conv = Conversation()
        if self.prompt and self.text:
            conv.add_message('user', self.prompt.format(text=self.text))
        elif self.text:
            conv.add_message('user', self.text)
            
        if self.responses:
            conv.add_message('assistant', self.responses[-1])
        return conv
    
    def to_dict(self, include_relations: bool = False):
        """
        Converts the instance into a dictionary.

        :param bool include_relations: Whether to include parent/child relationships.
        :return dict: A dictionary representation of the instance.
        """
        exclude_keys = {"parents", "children"} if not include_relations else set()
        exclude_keys.add("metadata")
        
        result = {
            k: copy.deepcopy(v)
            for k, v in self._data.items()
            if k not in exclude_keys
        }
        
        if include_relations:
            result["children"] = [child.to_dict(include_relations=False) 
                                 for child in self.children]
        
        return result

    @property
    def num_responses(self):
        """
        Returns the number of responses.

        :return int: The count of responses.
        """
        return len(self.responses)

    @property
    def num_labels(self):
        """
        Returns the number of labels.

        :return int: The count of labels.
        """
        return len(self.labels)

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        if name == '_data':
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return f"Instance({self._data.__repr__()})"

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __iter__(self):
        return self._data.__iter__()


if __name__ == '__main__':
    instance1 = Instance(text='test', prompt='test_prompt')
    instance2 = Instance(
        text='test', 
        prompt='test_prompt',
        responses=['response1', 'response2'],
        labels=[1, 0]
    )
    instance2.parents.append(instance1)
    instance1.children.append(instance2)

    print(instance1.to_dict())
    print(instance2.to_dict())