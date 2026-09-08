from dataclasses import dataclass, fields, is_dataclass
import importlib
from types import UnionType
from typing import get_args, get_origin

from reasoning_core.template import (
    Config,
    stochastic_rounding as sround,
)
from reasoning_core.tasks._procedural_warmup import ProceduralWarmupConfig
from reasoning_core.tasks.arithmetics import ArithmeticsConfig


MIGRATED_CONFIGS = [
    ("reasoning_core.tasks._procedural_warmup", "ProceduralWarmupConfig"),
    ("reasoning_core.integrations.reasoning_gym", "RGConfig"),
    ("reasoning_core.tasks.arithmetics", "ArithmeticsConfig"),
    ("reasoning_core.tasks.arithmetics", "WordProblemMathConfig"),
    ("reasoning_core.tasks.binding", "LambdaReductionConfig"),
    ("reasoning_core.tasks.binding", "RewriteSystemConfig"),
    ("reasoning_core.tasks.causal_reasoning", "Rung12Config"),
    ("reasoning_core.tasks.code_execution", "MesopyCodeCfg"),
    ("reasoning_core.tasks.code_execution", "CodeInputDeductionCfg"),
    ("reasoning_core.tasks.code_program_synthesis", "ProgramSynthesisCfg"),
    ("reasoning_core.tasks.constraint_satisfaction", "ConstraintSatisfactionConfig"),
    ("reasoning_core.tasks.coreference", "CoreferenceConfig"),
    ("reasoning_core.tasks.equation_system", "EquationSystemCfg"),
    ("reasoning_core.tasks.formal_analogies", "AnalogicalCaseMatchingConfig"),
    ("reasoning_core.tasks.game_playing", "GameBestMoveConfig"),
    ("reasoning_core.tasks.grammar", "GrammarConfig"),
    ("reasoning_core.tasks.grammar", "ParsingDerivationConfig"),
    ("reasoning_core.tasks.grammar", "ConstrainedContinuationConfig"),
    ("reasoning_core.tasks.grammar", "StressContinuationConfig"),
    ("reasoning_core.tasks.graph_operations", "GraphReasoningConfig"),
    ("reasoning_core.tasks.graph_operations", "GraphSuccessorsConfig"),
    ("reasoning_core.tasks.graph_operations", "GraphDependenciesConfig"),
    ("reasoning_core.tasks.logic_depth", "MultistepNLIConfig"),
    ("reasoning_core.tasks.logic_depth", "MultistepAbductionConfig"),
    ("reasoning_core.tasks.logic_semantics", "LogicConfig"),
    ("reasoning_core.tasks.math_geometry", "PlanarGeometryRelationsConfig"),
    ("reasoning_core.tasks.math_lean", "LeanConfig"),
    ("reasoning_core.tasks.math_metamath", "MetamathConfig"),
    ("reasoning_core.tasks.deprecated.math_rocq", "RocqConfig"),
    ("reasoning_core.tasks.math_tptp", "EntailConfig"),
    ("reasoning_core.tasks.math_tptp", "ConsistencyRepairConfig"),
    ("reasoning_core.tasks.grid_navigation", "GridNavigationConfig"),
    ("reasoning_core.tasks.planning", "PlanningConfig"),
    ("reasoning_core.tasks.probabilistic_reasoning", "MostProbableEvidenceConfig"),
    ("reasoning_core.tasks.probabilistic_reasoning", "MostProbableOutcomeConfig"),
    ("reasoning_core.tasks.qstr", "QualitativeReasoningConfig"),
    ("reasoning_core.tasks.qualitative_causal_reasoning", "QualitativeCausalReasoningConfig"),
    ("reasoning_core.tasks.regex", "RegexConfig"),
    ("reasoning_core.tasks.regex", "RegexRetrievalConfig"),
    ("reasoning_core.tasks.regex", "RegexReasoningConfig"),
    ("reasoning_core.tasks.sequential_induction", "SequenceConfig"),
    ("reasoning_core.tasks.set_operations", "SetOpsConfig"),
    ("reasoning_core.tasks.set_operations", "SetMissingElementConfig"),
    ("reasoning_core.tasks.set_operations", "CountElementsConfig"),
    ("reasoning_core.tasks.set_operations", "SetExpressionConfig"),
    ("reasoning_core.tasks.string_transduction", "StringTransductionConfig"),
    ("reasoning_core.tasks.table_qa", "TableQAConfig"),
    ("reasoning_core.tasks.table_qa", "TableStatisticsConfig"),
    ("reasoning_core.tasks.tracking", "ReferenceTrackingConfig"),
]


def test_stochastic_rounding_matches_config_template_rule():
    assert sround(3.0) == 3
    assert sround(3.25) in {3, 4}


def test_config_does_not_implicitly_round_integer_fields():
    @dataclass
    class PlainConfig(Config):
        n: int = 1

    config = PlainConfig()
    config.n = 1.5

    assert config.n == 1.5


def test_arithmetics_apply_difficulty_target_values():
    config = ArithmeticsConfig(seed=0).set_level(2)

    assert config.min_depth == 5
    assert config.max_depth == 7
    assert config.out_digits == 8
    assert config.out_decimals == 5


def test_procedural_warmup_apply_difficulty_scales_k():
    config = ProceduralWarmupConfig(seed=0).set_level(2)

    assert config.seq_len == 48
    assert config.vocab_size == 116
    assert config.k == 5
    assert config.max_depth == 1_000_000_004


def test_migrated_configs_have_explicit_apply_difficulty():
    for module_name, class_name in MIGRATED_CONFIGS:
        config_cls = getattr(importlib.import_module(module_name), class_name)
        assert "apply_difficulty" in config_cls.__dict__, f"{module_name}.{class_name}"


def test_migrated_configs_set_level_smoke():
    for module_name, class_name in MIGRATED_CONFIGS:
        config_cls = getattr(importlib.import_module(module_name), class_name)
        config = config_cls()
        for level in range(3):
            config.set_level(level)
            assert config.level == level
            config.to_dict()


def _matches_annotation(value, annotation):
    if annotation is None:
        return value is None
    if isinstance(annotation, str):
        return True
    origin = get_origin(annotation)
    if origin is None:
        if annotation is int:
            return isinstance(value, int) and not isinstance(value, bool)
        if annotation is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if annotation is bool:
            return isinstance(value, bool)
        if annotation is tuple:
            return isinstance(value, tuple)
        if annotation is list:
            return isinstance(value, list)
        if annotation is dict:
            return isinstance(value, dict)
        if annotation is str:
            return isinstance(value, str)
        if isinstance(annotation, type):
            return isinstance(value, annotation)
        return True
    if origin in (UnionType, getattr(__import__("typing"), "Union")):
        return any(_matches_annotation(value, arg) for arg in get_args(annotation))
    if origin in (tuple, list, dict):
        return isinstance(value, origin)
    return True


def test_migrated_configs_keep_annotated_field_types():
    for module_name, class_name in MIGRATED_CONFIGS:
        config_cls = getattr(importlib.import_module(module_name), class_name)
        if not is_dataclass(config_cls):
            continue
        for level in range(6):
            config = config_cls()
            config.set_level(level)
            for field in fields(config):
                value = getattr(config, field.name)
                if value is None and field.default is None:
                    continue
                assert _matches_annotation(value, field.type), (
                    f"{module_name}.{class_name}.{field.name} has value "
                    f"{value!r} of type {type(value).__name__}, expected {field.type}"
                )


def test_legacy_update_fallback_still_supported():
    @dataclass
    class LegacyConfig(Config):
        n: int = 1

        def update(self, c):
            self.n += c

    config = LegacyConfig()
    config.set_level(2)

    assert config.level == 2
    assert config.n == 3
