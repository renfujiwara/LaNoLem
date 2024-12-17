# LaNoLem

Code for reproducing [LaNoLem](https://arxiv.org/abs/2412.08114),

Gurobi optimizer is used in one of the comparison methods (MIOSR), so if you want to reproduce the evaluation experiments, you will need a Gurobi license (free for academic use).

## Quick demo (for poetry)
    Poetry install (if need)
    # Linux, macOS, Windows (WSL)
    $ curl -sSL https://install.python-poetry.org | python3 -
    
    # Windows (Powershell)
    $ (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

    Installation
    $ poetry install

    Quick demo: System Identification
    # (Please see Section 4)  
    See Demo.ipynb

