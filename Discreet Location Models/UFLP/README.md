### (UFLP) Uncapacitated Fixed-Charge Location Problem
- Find the number and location of facilities to minimize the total costs.
- Costs include the fixed charge of facilities and transportation costs of customers.
- Sets:
    - $I$: set of customers
    - $J$: set of candidate facility locations
- Parameters:
    - $c_{ij}$: cost to travel between customer $i$ and candidate facility at $j$
    - $h_i$: demand at customer $i$ (annual)
    - $f_j$: fixed charge at location $j$ (annual)
    - $\alpha$: transportation cost
- Decision variables:
    - $x_j=\begin{cases}1, \quad\text{if a facility is installed at } j\\0, \quad\text{O.W.}\end{cases}$
    - $y_{ij}=\begin{cases}1, \quad\text{if customer } i \text{ is covered by a facility at } j\\0, \quad\text{O.W.}\end{cases}$
        
<br>

$$
\begin{split}
\text{(UFLP)}\quad & \text{minimize}\quad \sum\limits_{j\in J}f_{j}x_{j} + \alpha\sum\limits_{i\in I}\sum\limits_{j\in J}h_{i}c_{ij}y_{ij}\\
&\begin{split}
\text{subject to}\quad \sum\limits_{j\in J}&y_{ij}=1 & \forall i\in I\\
    & y_{ij}\le x_{j}                & \forall i\in I, \forall j\in J\\
    & x_{j}\in \{0, 1\}              & \forall j\in J\\
    & y_{ij}\in \{0, 1\}\quad\quad   & \forall i\in I, \forall j\in J\\
\end{split}
\end{split}
$$
