### Shuntingyard
Lightweight parser for mathematical expressions, written in Python to avoid the need to use eval().
Uses Dijkstra's stack based infix-to-postfix translation algorithm.

#### Features
Parses and evaluates expressions containing numbers, variables, parenthesis, standard mathematical operators +, -, *, /, **, unary +/-, functions like sin(x) and SI prefixes like pico and kilo (p and k). Supports vector operations for Numpy array processing.

#### Dependencies
The only dependency is Numpy.

#### Provides
Shuntingyard provides a single class: ExpressionParser.

### Description

```python
from Shuntingyard import Expressionparser as Parser
p = Parser()
```

