#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 22:02:44 2026

@author: Marcel Hesselberth
"""


import re
import numpy as np


class ExpressionParser:
    def __init__(self, variables=None):
        self.variables = variables or {}
        # SI Prefixes
        self.prefixes = {12: 'T', 9: 'G', 6: 'M', 3: 'k', 0: '', -3: 'm', -6: 'u', -9: 'n', -12: 'p', -15: 'f'}
        self.p_map = {v: 10**k for k, v in self.prefixes.items() if v}
        self.p_chars = "".join(self.p_map.keys())

        # Ops: (Precedence, Function)
        self.ops = {
            '+': (1, np.add), '-': (1, np.subtract), '*': (2, np.multiply),
            '/': (2, np.divide), '**': (3, np.power), 'u-': (4, np.negative), 'u+': (4, lambda x: x)
        }
        # Funcs: (#args, f)
        self.funcs = {
            'sin': (1, np.sin), 'cos': (1, np.cos), 'exp': (1, np.exp),
            'abs': (1, np.abs), 'sqrt': (1, np.sqrt), 'sign': (1, np.sign),
            'pow': (2, np.pow),
        }

    def tokenize(self, expr):
        if not expr.strip(): return []
        # Group number-like sequences (including letters/signs immediately after digits)
        return re.findall(r'[\d\.][\d\.eE\+\-a-zA-Z]*|[a-zA-Z_]\w*|\*\*|[+\-*/(),]', expr)

    def to_postfix(self, expr_str):
        if not expr_str.strip(): raise ValueError("Empty expression")
        tokens = self.tokenize(expr_str)
        if not tokens: raise ValueError("Empty expression")

        output, op_stack, last_was_operand = [], [], False

        for tok in tokens:
            # 1. Number recognition with prefix validation
            if re.match(r'^[\d\.]', tok):
                if last_was_operand: raise ValueError(f"Expected operator before '{tok}'")
                
                # Regex for: [number] OR [number][prefix] OR [scientific]
                # STRICT: No prefix allowed after 'e' (e.g., 1e-3m is rejected)
                num_match = re.match(r'^(\d*\.?\d+)([' + self.p_chars + r'])?$', tok)
                sci_match = re.match(r'^\d*\.?\d+[eE][+-]?\d+$', tok)

                if num_match:
                    base, pref = num_match.groups()
                    output.append(('num', float(base) * self.p_map.get(pref, 1)))
                elif sci_match:
                    output.append(('num', float(tok)))
                else:
                    # Catch malformed numbers like 1.2.3, 10x, 1e-3m
                    raise ValueError(f"Invalid number or unknown variable: '{tok}'")
                
                last_was_operand = True
                continue

            # 2. Functions & Parentheses
            if tok in self.funcs:
                if last_was_operand: raise ValueError(f"Expected operator before '{tok}'")
                op_stack.append(('func', tok))
            elif tok == '(':
                if last_was_operand: raise ValueError(f"Expected operator before '('")
                op_stack.append(('paren', '('))
            elif tok == ',':
                while op_stack and op_stack[-1] != ('paren', '('):
                    output.append(op_stack.pop())
                if not op_stack: raise ValueError("Comma outside of function arguments")
                last_was_operand = False
            elif tok == ')':
                while op_stack and op_stack[-1] != ('paren', '('):
                    output.append(op_stack.pop())
                if not op_stack: raise ValueError("Mismatched parentheses")
                op_stack.pop() # '('
                if op_stack and op_stack[-1][0] == 'func':
                    output.append(op_stack.pop())
                last_was_operand = True

            # 3. Operators
            elif tok in self.ops or tok in '+-':
                if not last_was_operand:
                    if tok == '-': op_stack.append(('op', 'u-'))
                    elif tok == '+': op_stack.append(('op', 'u+'))
                    else: raise ValueError(f"Missing left operand for '{tok}'")
                else:
                    prec = self.ops[tok][0]
                    while op_stack and op_stack[-1][0] == 'op':
                        top_op = op_stack[-1][1]
                        top_prec = self.ops[top_op][0]
                        if (tok != '**' and top_prec >= prec) or (tok == '**' and top_prec > prec):
                            output.append(op_stack.pop())
                        else: break
                    op_stack.append(('op', tok))
                last_was_operand = False
            
            # 4. Variables
            else:
                if last_was_operand: raise ValueError(f"Expected operator before '{tok}'")
                output.append(('var', tok))
                last_was_operand = True

        while op_stack:
            op = op_stack.pop()
            if op == ('paren', '('): raise ValueError("Mismatched parentheses")
            output.append(op)
        
        if not output: raise ValueError("Incomplete expression")
        return output

    def evaluate(self, postfix, t_array):
        if not postfix: raise ValueError("Empty expression")
        stack = []
        ctx = {**self.variables, 't': t_array, 'pi': np.pi, 'e': np.e}
        
        for typ, val in postfix:
            if typ == 'num':
                stack.append(np.full_like(t_array, val, dtype=float))
            elif typ == 'var':
                if val not in ctx: raise ValueError(f"Unknown variable: '{val}'")
                v = ctx[val]
                stack.append(v if isinstance(v, np.ndarray) else np.full_like(t_array, float(v)))
            elif typ == 'op':
                if val in ('u-', 'u+'):
                    if not stack: raise ValueError(f"Missing operand for unary '{val[1]}'")
                    stack.append(self.ops[val][1](stack.pop()))
                else:
                    if len(stack) < 2: raise ValueError(f"Insufficient value(s) for operator '{val}'")
                    b, a = stack.pop(), stack.pop()
                    stack.append(self.ops[val][1](a, b))
            elif typ == 'func':
                n_args, func = self.funcs[val]
                if len(stack) < n_args: raise ValueError(f"Missing argument(s) for function '{val}'")
                args = [stack.pop() for _ in range(n_args)][::-1]
                stack.append(func(*args))

        if len(stack) != 1: raise ValueError("Invalid expression")
        return stack[0]
