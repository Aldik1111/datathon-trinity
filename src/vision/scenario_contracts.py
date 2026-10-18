"""Stable table-scenario identifiers shared by model output and its validator."""

SCENARIO_TYPES = frozenset({
    'cash_bag_unattended',
    'cash_count_missing',
    'client_left_before_count_complete',
    'door_left_open',
    'entry_through_open_door',
    'identity_check_missing',
    'mixed_client_cash',
    'mixed_denominations',
    'money_not_secured',
    'money_not_secured_before_client_left',
    'money_returned',
    'parallel_service',
    'personal_items_before_shift',
    'protected_items_unsecured',
    'report_record_missing',
    'safe_key_left',
    'safe_left_open',
    'unattended_screen',
})

ANALYSIS_CALLABLES = {
    'safe_state': 'src.vision.safe_classifier:analyze_video',
    'bank_scenarios': 'src.vision.scenario_classifier:analyze_video',
}
