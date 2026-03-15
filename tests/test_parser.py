#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:54:20 2026

@author: Marcel Hesselberth
"""

import pytest
import numpy as np
import sys, os
sys.path.append(os.getcwd() + '/..')
from parser import ExpressionParser


# ───────────────────────────────────────────────
# Fixtures & helpers
# ───────────────────────────────────────────────

@pytest.fixture
def variables():
    return {
        'Vcc': 5.0,
        'Vdd': 12.0,
        'tau': 0.01,
        'f': 50.0,
        'omega': 100.0,
        'phase': 0.0,
        'pi': np.pi,
        'e': np.e,
    }


@pytest.fixture
def parser(variables):
    return ExpressionParser(variables=variables)


@pytest.fixture
def t_small():
    return np.array([0.0, 0.002, 0.005, 0.01])


@pytest.fixture
def t_empty():
    return np.array([])


RTOL = 1e-9
ATOL = 1e-12


def assert_array_close(actual, expected):
    if np.isscalar(expected):
        expected = np.full_like(actual, expected)
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


# ───────────────────────────────────────────────
# Positive tests – should succeed
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_at_0", [
    ("1+1", 2.0),
    ("2 + 3", 5.0),
    ("-5 + 7.5", 2.5),
    ("2 * -3", -6.0),
    ("2 ** 3 ** 2", 512.0),
    ("10/5 / 2", 1.0),
    ("- - - -8", 8.0),
    ("- ---8", 8.0),
    ("sin(pi/2)", 1.0),
    ("abs(-3.2)", 3.2),
    ("sqrt(16)", 4.0),
    ("pow(3, 4)", 81.0),
    ("Vcc * (1 - exp(-t / tau))", 0.0),
    ("1.23e-4 + 5", 5.000123),
    ("-4.5e+2 * 2", -900.0),
    (".5e-1", 0.05),
    ("-sqrt(16)", -4.0),
    ("--abs(-5)", 5.0),
])
def test_correct_expressions(parser, t_small, expr, expected_at_0):
    parser.parse(expr)
    result = parser.eval(t_small)
    assert_array_close(result[0], expected_at_0)


# ───────────────────────────────────────────────
# Error cases 
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_exc_msg_substring", [
    # Mismatched parentheses
    ("(1 + 2",           "Mismatched parentheses"),
    ("1 + 2)",           "Mismatched parentheses"),
    ("((3 * 4)",         "Mismatched parentheses"),
    (") + 5",            "Mismatched parentheses"),

    # Wrong number of function arguments
    ("sin()",            "Missing argument(s) for 'sin'"),
    ("sin(1,2)",         "Invalid expression"),
    ("pow(t)",           "Missing argument(s) for 'pow'"),
    ("pow(0.5)",      "Missing argument(s) for 'pow'"),
    ("pow()",   "Missing argument(s) for 'pow'"),

    # Syntax / missing operands
    ("3 + * 4",          "Missing left operand for '*'"),
    ("5 / / 2",          "Missing left operand for '/'"),
    ("2 3 + 4",          "Expected operator before '3'"),
    ("+ +",              "Missing operand for unary '+'"),

    # Unknown variables
    ("x + 5",            "Unknown variable: 'x'"),
    ("Vcc + unknown",    "Unknown variable: 'unknown'"),

    # Comma outside function (tends to cause stack or paren error)
    (", 5",              "Comma outside of function arguments"),
    ("5 + , 3",          "Comma outside of function arguments"),
    ("(1, 2)",           "Invalid expression"),

    # Malformed numbers — current tokenizer accepts many as variables → later fails
    ("1.2.3",            "Expected operator before '.3'"),
    ("1e",               "Expected operator before 'e'"),
    ("1e+",              "Expected operator before 'e'"),
    ("e4",               "Unknown variable: 'e4'"),
])
def test_error_cases_real_messages(parser, t_small, expr, expected_exc_msg_substring):
    with pytest.raises(ValueError) as exc_info:
        parser.parse(expr)
        _ = parser.eval(t_small)

    assert expected_exc_msg_substring in str(exc_info.value)


# ───────────────────────────────────────────────
# Numerical edge cases (should not raise — produce inf/nan)
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr", [
    "1 / 0",
    "sqrt(-1)",
    "1 / (t - t)",
    "exp(1000)",
])
def test_numerical_edges_no_exception(parser, t_small, expr):
    parser.parse(expr)
    result = parser.eval(t_small)
    assert len(result) == len(t_small)
    # We allow inf / nan
    assert np.all(np.isfinite(result) | np.isnan(result) | np.isinf(result))


# ───────────────────────────────────────────────
# Empty input array
# ───────────────────────────────────────────────

def test_empty_t_array(parser, t_empty):
    expr = "Vcc * sin(2 * pi * f * t + phase)"
    parser.parse(expr)
    result = parser.eval(t_empty)
    assert result.shape == (0,)
    assert result.dtype.kind == 'f'


def test_empty_expression(parser, t_small):
    with pytest.raises(ValueError):
        # Current code returns zeros_like → but many would prefer error
        # If you want to make it raise → change evaluate when postfix empty
        result = parser.eval([], t_small)
        assert np.all(result == 0)


# ───────────────────────────────────────────────
# Deep nesting (should usually work until stack/recursion limit)
# ───────────────────────────────────────────────

def test_deep_nesting(parser, t_small):
    # ~20 levels — should be fine
    expr = "sin(" * 10 + "t" + ")" * 10
    parser.parse(expr)
    result = parser.eval(t_small)
    assert len(result) == len(t_small)


# ───────────────────────────────────────────────
# Prefix tests – should succeed
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_val", [
    ("1k", 1000.0),
    ("100m", 0.1),
    ("1u", 1e-6),
    ("4.7k + 300", 5000.0),
    ("Vcc * 10m", 0.05),              # Vcc (5.0) * 0.01
    ("1M / 1k", 1000.0),              # Mega / kilo
    ("sin(1000m * pi / 2)", 1.0),     # 1000m is 1
    ("500u * 2k", 1.0),               # 0.0005 * 2000
    ("1e-3 * 1k", 1.0),               # Mix van scientific en prefix
    ("100n * 10**9", 100.0),          # nano
    ("1p * 1T", 1.0),                 # pico * Tera
    ("1f * 1e15", 1.0),               # femto
    ("-10k", -10000.0),               # Unaire minus met prefix
    ("abs(-100m)", 0.1),              # Functie met prefix
    ("pow(t, 1k)", 0.0),              # Meerdere argumenten met prefixes (bij t=0)
])
def test_prefixes_correct(parser, t_small, expr, expected_val):
    parser.parse(expr)
    result = parser.eval(t_small)
    # Check de eerste waarde (t=0)
    assert_array_close(result[0], expected_val)


@pytest.mark.parametrize("expr, expected_at_0", [
    ("1e-3", 0.001),                  # Scientific notation (geen prefix '3')
    ("1m", 0.001),                    # Prefix notation
])
def test_scientific_vs_prefix(parser, t_small, expr, expected_at_0):
    # Deze test garandeert dat 'e' in 1e3 niet als variabele of prefix wordt gezien
    parser.parse(expr)
    result = parser.eval(t_small)
    assert_array_close(result[0], expected_at_0)


# ───────────────────────────────────────────────
# Prefix Error cases
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_exc_msg_substring", [
    ("10x", "Expected operator before 'x'"),      # 'x' is geen prefix
    ("10 k", "Expected operator before 'k'"), # Spatie tussen getal en prefix (als k een var is)
    ("k10", "Unknown variable: 'k10'"),      # Prefix voor het getal
    ("1mk", "Expected operator before 'k'"),      # Dubbele prefix (niet toegestaan)
    ("1e-3m", "Expected operator before 'm'"),  # Prefix direct na exponent zonder spatie
    ("1.0e3k", "Expected operator before 'k'"),  # Prefix direct na exponen
])
def test_prefix_errors(parser, t_small, expr, expected_exc_msg_substring):
    with pytest.raises(ValueError) as exc_info:
        parser.parse(expr)
        _ = parser.eval(t_small)
    assert expected_exc_msg_substring in str(exc_info.value)


# ───────────────────────────────────────────────
# Extra complexe expressies (Prefix + Functies)
# ───────────────────────────────────────────────

def test_complex_prefix_expression(parser, t_small):
    # Expressie: Vcc (5) * sin(2 * pi * 50 * t) + 100m
    # Bij t=0: 5 * sin(0) + 0.1 = 0.1
    expr = "Vcc * sin(2 * pi * 50 * t) + 100m"
    parser.parse(expr)
    result = parser.eval(t_small)
    assert_array_close(result[0], 0.1)

def test_prefix_in_pow(parser):
    # t=1000, exp=0.001
    t = np.array([1000, 0.001]) 
    expr = "pow(t, 0.01f)" 
    parser.parse(expr)
    result = parser.eval(t)
    assert result[0] == 1.0

# ───────────────────────────────────────────────
# Exponentiation & Unary tests – should succeed
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_at_0", [
    # Right-associativity of ** (2^(3^2))
    ("2 ** 3 ** 2", 512.0),
    ("3 ** 2 ** 3", 6561.0),
    
    # Power vs Unary binding: -2**2 should be -(2**2) = -4
    ("-2 ** 2", -4.0),
    ("(-2) ** 2", 4.0),
    ("-3 ** -2", -0.1111111111111111), # -(3^-2)
    
    # Multiple unaries
    ("---5", -5.0),
    ("-+-5", 5.0),
    ("2 * - - 3", 6.0),
    
    # Complex combinations
    ("2 ** -3 ** 2", 1.953125e-03), # 2**(-(3**2)) = 2^-9
    ("10 ** 3 / 1k", 1.0),
    ("-(2+3)**2", -25.0),
])
def test_exponentiation_and_unaries(parser, t_small, expr, expected_at_0):
    parser.parse(expr)
    result = parser.eval(t_small)
    assert_array_close(result[0], expected_at_0)


# ───────────────────────────────────────────────
# Error cases – Exponentiation & Unaries
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_exc_msg_substring", [
    # Missing operands for power
    ("** 3",              "Missing left operand for '**'"),
    ("2 **",              "Missing operand for '**'"),
    ("2 ** * 3",          "Missing left operand for '*'"),
    
    # Incomplete unaries
    ("5 + -",             "Missing operand for unary '-'"),
    ("3 * +",             "Missing operand for unary '+'"),
    
    # Binary operators used as unary (other than +/-)
    ("* 5",               "Missing left operand for '*'"),
    ("/ 10",              "Missing left operand for '/'"),
    
    # Invalid combinations
    ("2 ** ( )",          "Missing operand for '**'"),
    ("pow(2, **3)",       "Missing left operand for '**'"),
])
def test_error_cases_ops(parser, t_small, expr, expected_exc_msg_substring):
    with pytest.raises(ValueError) as exc_info:
        parser.parse(expr)
        _ = parser.eval(t_small)
    assert expected_exc_msg_substring in str(exc_info.value)

# ───────────────────────────────────────────────
# Prefix & Exponentiation – should succeed
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_at_0", [
    ("1k ** 2", 1000000.0),           # (1000)^2
    ("10m ** 3", 1e-6),              # (0.01)^3
    ("2u ** -1", 500000.0),           # 1 / (2e-6)
    
    # Combinations with other ops
    ("1k ** 2 / 1M", 1.0),            # 1,000,000 / 1,000,000
    ("2 * 10k ** 2", 200000000.0),    # 2 * (10000^2)
    
    # Nested/Complex
    ("(2k)**2", 4000000.0),
    ("100m ** (1+1)", 0.01),          # (0.1)^2
    
    # Prefixes in both base and exponent (hoewel exponenten met prefix zeldzaam zijn)
    ("2 ** 1000m", 2.0),              # 2^1 (1000m = 1)
    ("4 ** 500m", 2.0),               # 4^0.5 = sqrt(4)
])
def test_prefix_exponentiation(parser, t_small, expr, expected_at_0):
    parser.parse(expr)
    result = parser.eval(t_small)
    assert_array_close(result[0], expected_at_0)

# ───────────────────────────────────────────────
# Error cases – Prefix & Exponents
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_exc_msg_substring", [
    # Malformed prefixes near operators
    ("1k* *2",            "Missing left operand for '*'"), # Spatie tussen **
    ("1.k ** 2",          "Expected operator before 'k'"),
    ("k1 ** 2",           "Unknown variable: 'k1'"),       # Prefix moet achter getal
])
def test_error_prefix_ops(parser, t_small, expr, expected_exc_msg_substring):
    with pytest.raises(ValueError) as exc_info:
        parser.parse(expr)
        _ = parser.eval(t_small)
    assert expected_exc_msg_substring in str(exc_info.value)
    
# ───────────────────────────────────────────────
# Negative Prefixes & Exponents – should succeed
# ───────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected_at_0", [
    ("-10k ** 2", -100000000.0),      # -(10000^2) = -1e8 (want macht bindt sterker)
    ("(-10k) ** 2", 100000000.0),     # (-10000)^2 = 1e8
    ("-2m ** 3", -8e-9),              # -(0.002^3) = -8e-9
    ("(-2m) ** 3", -8e-9),             # (-0.002)^3 = -8e-9
    # Combinaties met breuken en prefixes
    ("-1u ** -2", -1e12),   # -1e12.
    # Prefix in de exponent bij negatieve basis
    ("(-2) ** 2000m", 4.0),           # (-2)^2 = 4
])
def test_negative_prefix_exponentiation(parser, t_small, expr, expected_at_0):
    parser.parse(expr)
    result = parser.eval(t_small)
    assert_array_close(result[0], expected_at_0)