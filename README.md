# Shuntingyard ExpressionParser

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![NumPy](https://img.shields.io/badge/dependency-numpy-orange)
![License](https://img.shields.io/badge/license-GPLv3-green)
![Status](https://img.shields.io/badge/status-stable-brightgreen)

A **NumPy‑powered mathematical expression parser and evaluator**
implemented in Python.

`ExpressionParser` converts human‑readable mathematical expressions into
**postfix notation (Reverse Polish Notation)** using a Shunting‑Yard
style algorithm and evaluates them using **vectorized NumPy
operations**.

It supports:

-   arithmetic operators
-   unary operators
-   variables
-   built‑in and user‑defined functions
-   NumPy vectorized evaluation
-   engineering **SI prefixes**
-   **scientific notation**
-   scalar **and array evaluation**

This makes it useful for:

-   scientific computing
-   engineering tools
-   signal processing
-   simulation pipelines
-   embedded scripting languages

------------------------------------------------------------------------

# Table of Contents

-   [Installation](#installation)
-   [Quick Example](#quick-example)
-   [Features](#features)
-   [Supported Syntax](#supported-syntax)
-   [Operators](#operators)
-   [Built‑in Functions](#built-in-functions)
-   [User‑Defined Functions](#user-defined-functions)
-   [Variables](#variables)
-   [Default Variable Behavior](#default-variable-behavior)
-   [SI Prefix Support](#si-prefix-support)
-   [Scientific Notation](#scientific-notation)
-   [Architecture](#architecture)
-   [Evaluation Model](#evaluation-model)
-   [Error Handling](#error-handling)
-   [Verbose Debugging](#verbose-debugging)
-   [Examples](#examples)

------------------------------------------------------------------------

# Installation

Dependencies:

    numpy

Install via pip:

    pip install numpy

------------------------------------------------------------------------

# Quick Example

``` python
from parser import ExpressionParser
import numpy as np

p = ExpressionParser()

p.parse("2 + 3 * 4")
print(p.eval())
```

Output

    14

------------------------------------------------------------------------

# Features

✔ Shunting‑yard expression parsing\
✔ postfix stack evaluator\
✔ NumPy vectorized execution\
✔ unary operator detection\
✔ exponentiation support\
✔ engineering prefixes (k, M, m, µ...)\
✔ scientific notation\
✔ user‑defined functions\
✔ scalar or array evaluation

------------------------------------------------------------------------

# Supported Syntax

Example expressions:

    2+3*4
    sin(x)
    cos(pi-sin(x))
    sqrt(x**2 + y**2)
    exp(-t/5)
    atan2(y,x)
    10k + 3M

------------------------------------------------------------------------

# Operators

  Operator   Meaning          Example
  ---------- ---------------- ---------
  \+         addition         `a+b`
  \-         subtraction      `a-b`
  \*         multiplication   `a*b`
  /          division         `a/b`
  \*\*       exponentiation   `2**3`
  u-         unary minus      `-x`
  u+         unary plus       `+x`

Operator precedence:

    **   highest
    unary + -
    *  /
    +  -

Exponentiation is **right associative**.

Example

    2**3**2 = 2**(3**2)

------------------------------------------------------------------------

# Built-in Functions

  Function   Description
  ---------- ----------------
  sin(x)     sine
  cos(x)     cosine
  exp(x)     exponential
  abs(x)     absolute value
  sqrt(x)    square root
  sign(x)    sign function
  pow(a,b)   power

Example

    sin(pi/2)
    sqrt(16)
    pow(2,3)

------------------------------------------------------------------------

# User Defined Functions

Users can extend the parser at runtime.

    add_function(name, function, argument_count)

Example

``` python
import numpy as np

p.add_function("atan2", np.atan2, 2)

p.parse("atan2(y,x)")
```

------------------------------------------------------------------------

# Variables

Variables are supplied through the parser context.

    ExpressionParser(variables={})

Example

``` python
p = ExpressionParser({'x':5,'y':2})

p.parse("x+y")
p.eval()
```

Result

    7

Variables may be:

-   scalars
-   NumPy arrays

Example

``` python
x = np.linspace(0,10,100)

p = ExpressionParser({'x':x})

p.parse("sin(x)")
p.eval()
```

Result

    array of 100 values

------------------------------------------------------------------------

# Default Variable Behavior

The parser supports a **default variable** (default name: `t`).

This variable enables automatic **vector broadcasting**.

Example:

    default_var = 't'

If the user supplies:

    t = array length L

and evaluates a constant expression:

    5

The parser produces:

    array([5,5,5,...]) length L

Example:

``` python
import numpy as np

t = np.linspace(0,10,100)

p = ExpressionParser(default_var='t')

p.parse("5")

result = p.eval(var_array=t)
```

Output

    array([5,5,5,...]) length 100

This behavior ensures **all results are immediately compatible with
NumPy pipelines** without manual broadcasting.

Example:

    5 + sin(t)

Both operands automatically become arrays.

------------------------------------------------------------------------

# SI Prefix Support

Engineering prefixes are supported.

  Prefix   Value
  -------- -------
  T        1e12
  G        1e9
  M        1e6
  k        1e3
  m        1e-3
  u        1e-6
  n        1e-9
  p        1e-12
  f        1e-15

Example

    10k = 10000
    3M = 3000000
    5m = 0.005

------------------------------------------------------------------------

# Scientific Notation

Scientific notation is supported.

    1e3
    2.5e-6
    3E8

Example

    1e3 + 2e3 = 3000

------------------------------------------------------------------------

# Architecture

    Expression string
          │
          ▼
    Tokenizer (regex)
          │
          ▼
    Shunting‑Yard Parser
          │
          ▼
    Postfix Program
          │
          ▼
    Stack Evaluator
          │
          ▼
    NumPy Result

------------------------------------------------------------------------

# Evaluation Model

Postfix stack execution.

Example

Expression

    3 + 4*2

Postfix

    3 4 2 * +

Stack execution

    push 3
    push 4
    push 2
    multiply
    add

Result

    11

------------------------------------------------------------------------

# Error Handling

The parser performs extensive validation.

### Empty expression

    p.parse("")

Error

    ValueError: Empty expression

------------------------------------------------------------------------

### Missing operand

    p.parse("3+")

Error

    ValueError: Missing operand for '+'

------------------------------------------------------------------------

### Missing left operand

    p.parse("*5")

Error

    ValueError: Missing left operand for '*'

------------------------------------------------------------------------

### Mismatched parentheses

    p.parse("(3+4")

Error

    ValueError: Mismatched parentheses

------------------------------------------------------------------------

### Comma outside function

    p.parse("3,4")

Error

    ValueError: Comma outside of function arguments

------------------------------------------------------------------------

### Unknown variable

    p.parse("x+1")
    p.eval()

Error

    ValueError: Unknown variable: 'x'

------------------------------------------------------------------------

### Invalid numbers

    p.parse("1.2.3")

Error

    ValueError: Invalid number or unknown variable

------------------------------------------------------------------------

### Missing function arguments

    p.parse("pow(2)")

Error

    ValueError: Missing argument(s) for 'pow'

------------------------------------------------------------------------

# Verbose Debugging

Debug execution with:

``` python
p.eval(verbose=True)
```

Example output

    variables: {'x':5}

    [] 3
    [3] 4
    [3,4] +
    result 7

Shows stack evolution step‑by‑step.

------------------------------------------------------------------------

# Examples

## Scalar evaluation

``` python
p = ExpressionParser()

p.parse("2+3*4")
p.eval()
```

Result

    14

------------------------------------------------------------------------

## Variable evaluation

``` python
p = ExpressionParser({'x':10})

p.parse("x*2+5")
p.eval()
```

Result

    25

------------------------------------------------------------------------

## Vector evaluation

``` python
x = np.linspace(0,10,100)

p = ExpressionParser({'x':x})

p.parse("sin(x)")
y = p.eval()
```

------------------------------------------------------------------------

## Engineering notation

    p.parse("3k + 5M")

Result

    5003000

------------------------------------------------------------------------

# Summary

`ExpressionParser` is a compact yet powerful expression engine
providing:

-   robust mathematical parsing
-   NumPy‑accelerated evaluation
-   extensibility through custom functions
-   automatic broadcasting via the default variable

It is suitable for:

-   scientific scripting
-   engineering calculations
-   simulation frameworks
-   signal processing pipelines
-   custom mathematical DSLs
