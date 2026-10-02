"""Normalize single-conversation chat-template outputs across HF releases."""
from collections.abc import Mapping
from numbers import Integral


def chat_token_ids(tokenizer, text):
    ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=True,
        add_generation_prompt=True, return_dict=False)
    if isinstance(ids, Mapping):
        ids = ids['input_ids']
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    ids = list(ids)
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError('Expected exactly one tokenized conversation')
        ids = list(ids[0])
    if not ids or any(not isinstance(token, Integral) for token in ids):
        raise ValueError('Chat template did not return a nonempty sequence of token IDs')
    return [int(token) for token in ids]
