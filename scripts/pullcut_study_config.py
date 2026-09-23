"""Shared configuration for reproducible pull-threshold studies."""
import math
import re


def validate_inputs(inputs):
    names = tuple(inputs['scenarios'])
    selections = inputs['selections']
    profiles = inputs['profiles']
    groups = inputs['comparison_sets']
    for label in (*names, *selections, *profiles, *groups, inputs['study_tag']):
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', label):
            raise ValueError(f'Invalid study identifier: {label!r}')
    if not selections or not profiles or not groups:
        raise ValueError('Selections, profiles and comparison sets must be nonempty')
    for threshold in selections.values():
        if threshold is not None and (not math.isfinite(threshold) or threshold <= 0):
            raise ValueError('Pull thresholds must be finite and positive, or null for uncut')
    for key in ('particle_pt_min_gev',):
        value = inputs.get(key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{key} must be finite and positive')
    if len(profiles) != len(set(profiles)) or len(set(selections.values())) != len(selections):
        raise ValueError('Duplicate profiles or thresholds')
    for profile in profiles:
        if profile not in ('eta2-mll71-111', 'eta3-mll71-111', 'eta45-mll71-111'):
            raise ValueError(f'Unsupported pseudorapidity profile: {profile}')
    reference = inputs['reference_scenario']
    for members in groups.values():
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError('Comparisons need distinct scenarios')
        if members[0] != reference or not set(members).issubset(names):
            raise ValueError('Each comparison must start with the configured nominal scenario')
    if not any(set(members) == set(names) for members in groups.values()):
        raise ValueError('A complete comparison is required for pairwise closure checks')
    for key in ('workers', 'event_shards', 'concurrent_analyses', 'smoke_workers', 'smoke_events'):
        if not isinstance(inputs[key], int) or inputs[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not inputs['luminosities_fb'] or any(not math.isfinite(x) or x <= 0 for x in inputs['luminosities_fb']):
        raise ValueError('Luminosities must be finite and positive')
    if inputs['smoke_profile'] not in profiles or not set(inputs['smoke_selections']).issubset(selections):
        raise ValueError('Smoke settings must belong to the production study')
    for field, allowed in (('reuse_sources', names), ('reuse_comparisons', groups)):
        for profile, by_selection in inputs.get(field, {}).items():
            if profile not in profiles:
                raise ValueError(f'Unknown reused profile: {profile}')
            for selection, runs in by_selection.items():
                if selection not in selections or not set(runs).issubset(allowed):
                    raise ValueError('Unknown reused selection/scenario/comparison')
                if any('/' in value or value in ('.', '..') for value in runs.values()):
                    raise ValueError('Reused results must be immutable run identifiers')
    return inputs


def selection_text(threshold):
    return 'No pull-magnitude cut' if threshold is None else f'|t⃗| > {threshold:g} per measured jet'


def comparison_key(profile, selection, group):
    return f'{profile}/{selection}/{group}'


def analysis_label(inputs, name, profile, selection, smoke=False):
    return f'{name}-{profile}-{selection}-{inputs["study_tag"]}' + ('-smoke' if smoke else '')


def comparison_label(inputs, profile, selection, group, smoke=False):
    return f'{group}-{profile}-{selection}-{inputs["study_tag"]}' + ('-smoke' if smoke else '')


def event_arguments(inputs, name, profile, selection, smoke=False):
    arguments = ['--analyses', 'cutbased', '--luminosities', *map(str, inputs['luminosities_fb']),
                 '--samples', inputs['scenarios'][name]['manifest'], '--cut-profile', profile]
    if inputs.get('particle_pt_min_gev') is not None:
        arguments += ['--particle-pt-min', str(inputs['particle_pt_min_gev'])]
    threshold = inputs['selections'][selection]
    if threshold is not None:
        arguments += ['--pull-magnitude-min', str(threshold)]
    arguments += ['--workers', str(inputs['smoke_workers'] if smoke else inputs['workers']),
                  '--event-shards', str(inputs['smoke_workers'] if smoke else inputs['event_shards'])]
    if smoke:
        arguments += ['--max-events', str(inputs['smoke_events'])]
    return arguments


def comparison_arguments(inputs, group, runs):
    members = inputs['comparison_sets'][group]
    arguments = ['--compare-runs', *[str(runs[name]) for name in members], '--analyses', 'cutbased',
                 '--luminosities', *map(str, inputs['luminosities_fb'])]
    for name in members:
        scenario = inputs['scenarios'][name]
        if scenario.get('color'):
            arguments += ['--scenario-color', scenario['identifier'] + '=' + scenario['color']]
    return arguments
