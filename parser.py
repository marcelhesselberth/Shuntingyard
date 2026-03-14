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
        self.prefixes = {12: 'T', 9: 'G', 6: 'M', 3: 'k', 0: '', \
                         -3: 'm', -6: 'u', -9: 'n', -12: 'p', -15: 'f'}
        self.p_map = {v: 10**k for k, v in self.prefixes.items() if v}
        self.p_chars = "".join(self.p_map.keys())

        # Ops: (Precedence, Function). Power (4) > Unary (3) for math correctness.
        self.ops = {
            '+': (1, np.add), '-': (1, np.subtract), 
            '*': (2, np.multiply), '/': (2, np.divide), 
            'u-': (3, np.negative), 'u+': (3, lambda x: x),
            '**': (4, np.power)
        }
        # Funcs: (#args, f)
        self.funcs = {
            'sin': (1, np.sin), 'cos': (1, np.cos), 'exp': (1, np.exp),
            'abs': (1, np.abs), 'sqrt': (1, np.sqrt), 'sign': (1, np.sign),
            'pow': (2, np.power)
        }
        self.postfix = []


    def tokenize(self, expr):
        if not expr.strip(): return []
        # Pattern ensures SI prefixes are tied to numbers, otherwise they are variables
        pattern = r'\d*\.?\d+[eE][+-]?\d+|\d*\.?\d+[' + self.p_chars \
                + r']?|[a-zA-Z_]\w*|\*\*|[+\-*/(),]'
        return re.findall(pattern, expr)


    def parse(self, expr_str):
        if not expr_str.strip(): raise ValueError("Empty expression")
        tokens = self.tokenize(expr_str)
        if not tokens: raise ValueError("Empty expression")

        output, op_stack, last_was_operand = [], [], False

        for tok in tokens:
            # Number recognition
            if re.match(r'^[\d\.]', tok):
                if last_was_operand: raise ValueError(f"Expected operator before '{tok}'")
                
                if 'e' in tok.lower():
                    # Handle scientific notation and malformed cases like '1e'
                    try:
                        if tok.lower().endswith(('e', 'e+', 'e-')): raise ValueError()
                        output.append(('num', float(tok)))
                    except ValueError:
                        raise ValueError(f"Invalid number or unknown variable: '{tok}'")
                else:
                    # Handle numbers with/without SI prefixes or malformed '1.2.3'
                    num_match = re.match(r'^(\d*\.?\d+)([' + self.p_chars + r'])?$', tok)
                    if num_match:
                        base, pref = num_match.groups()
                        try:
                            val = float(base) * self.p_map.get(pref, 1)
                            output.append(('num', val))
                        except ValueError:
                            raise ValueError(f"Invalid number or unknown variable: '{tok}'")
                    else:
                        raise ValueError(f"Invalid number or unknown variable: '{tok}'")
                last_was_operand = True
                continue

            # Functions & Parentheses
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
                op_stack.pop() # Remove '('
                if op_stack and op_stack[-1][0] == 'func':
                    output.append(op_stack.pop())
                last_was_operand = True

            # Operators
            elif tok in self.ops:
                if not last_was_operand:
                    if tok == '-': op_stack.append(('op', 'u-'))
                    elif tok == '+': op_stack.append(('op', 'u+'))
                    else: raise ValueError(f"Missing left operand for '{tok}'")
                else:
                    prec = self.ops[tok][0]
                    while op_stack and op_stack[-1][0] == 'op':
                        top_op = op_stack[-1][1]
                        top_prec = self.ops[top_op][0]
                        # Correct Associativity check
                        if (tok == '**' and top_prec > prec) or (tok != '**' and top_prec >= prec):
                            output.append(op_stack.pop())
                        else: break
                    op_stack.append(('op', tok))
                last_was_operand = False
            
            # Variables
            else:
                if last_was_operand: raise ValueError(f"Expected operator before '{tok}'")
                output.append(('var', tok))
                last_was_operand = True

        # Check if expression ended with an operator
        if not last_was_operand and tokens:
            last_tok = tokens[-1]
            if last_tok in '+-':
                raise ValueError(f"Missing operand for unary '{last_tok}'")
            raise ValueError(f"Missing operand for '{last_tok}'")

        while op_stack:
            op = op_stack.pop()
            if op == ('paren', '('): raise ValueError("Mismatched parentheses")
            output.append(op)
        
        if not output: raise ValueError("Incomplete expression")
        self.postfix = output
        return output

    
    def eval(self, t_array, postfix=None):
        postfix = postfix or self.postfix
        if not postfix:
            raise ValueError("Empty expression")
        
        stack = []
        # t, pi en e zijn standaard beschikbaar; self.variables kan ze overschrijven
        ctx = {**self.variables, 't': t_array, 'pi': np.pi, 'e': np.e}
        
        for typ, val in postfix:
            if typ == 'num':
                stack.append(np.full_like(t_array, val, dtype=float))
            
            elif typ == 'var':
                if val not in ctx:
                    raise ValueError(f"Unknown variable: '{val}'")
                v = ctx[val]
                # Zorg dat variabelen altijd als array terugkomen voor NumPy operaties
                stack.append(v if isinstance(v, np.ndarray) else np.full_like(t_array, float(v)))
            
            elif typ == 'op':
                if val in ('u-', 'u+'):
                    if len(stack) < 1:
                        # Gebruik de laatste letter van de operatornaam (bijv. '-' uit 'u-')
                        raise ValueError(f"Missing operand for unary '{val[-1]}'")
                    stack.append(self.ops[val][1](stack.pop()))
                else:
                    if len(stack) < 2:
                        raise ValueError(f"Missing operand for '{val}'")
                    b, a = stack.pop(), stack.pop()
                    stack.append(self.ops[val][1](a, b))
            
            elif typ == 'func':
                n_args, func = self.funcs[val]
                if len(stack) < n_args:
                    raise ValueError(f"Missing argument(s) for '{val}'")
                
                # Haal argumenten van de stack en draai ze om (stack is LIFO)
                args = [stack.pop() for _ in range(n_args)][::-1]
                stack.append(func(*args))
    
        # Expression mathematically incomplete
        if len(stack) != 1:
            raise ValueError("Invalid expression")
            
        return stack[0]

if __name__ == "__main__":
    p = ExpressionParser()
    p.parse("1kk")
    print(p.eval(0))