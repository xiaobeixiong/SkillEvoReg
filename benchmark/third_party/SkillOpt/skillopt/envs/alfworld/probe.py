"""Semantics-preserving presentation perturbations, independent of the simulator."""
import hashlib
import json
import re

ATTACK_TYPES = ('admissible_action_order', 'action_menu_format', 'observation_layout')


def render_probe(observation, actions, attack_type=None, seed=None):
    """Change presentation only; never rewrite identifiers or executable commands."""
    if attack_type not in (None, *ATTACK_TYPES):
        raise ValueError(f'Unknown ALFWorld probe: {attack_type}')
    visible = [s for s in actions if s != 'help']
    presented = order_actions(visible, seed) if attack_type == 'admissible_action_order' else visible
    menu = '\n '.join(f"'{s}'" for s in presented)
    if attack_type == 'action_menu_format':
        menu = json.dumps(presented, ensure_ascii=False)
    displayed = observation
    if attack_type == 'observation_layout':
        # Preserve every non-whitespace token, its order, and all punctuation.
        displayed = re.sub(r'\s+', ' ', observation).strip()
    original_menu = '\n '.join(f"'{s}'" for s in visible)
    evidence = {'attack_type': attack_type, 'changed': displayed != observation or menu != original_menu,
                'observation_tokens_preserved': displayed.split() == observation.split(),
                'action_multiset_preserved': sorted(presented) == sorted(visible),
                'before_sha256': hashlib.sha256((observation+'\n'+original_menu).encode()).hexdigest(),
                'after_sha256': hashlib.sha256((displayed+'\n'+menu).encode()).hexdigest()}
    return displayed, menu, evidence


def order_actions(actions, seed=None):
    if seed is None:
        return list(actions)
    result = sorted(actions, key=lambda action: hashlib.sha256(f'{seed}:{action}'.encode()).digest())
    if result == list(actions) and len(set(result)) > 1:
        result = result[1:] + result[:1]
    return result
