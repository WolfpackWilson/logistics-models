### (NTP) Network Transportation Problem
- Find the shipment patterns from supply points to demand points such that all demands are satisfied and the total cost is minimized.
- Sets:
    - $I$: set of suppliers
    - $J$: set of demand points
- Parameters:
    - $c_{ij}$: unit cost of shipping from supply node $i\in I$ to demand node $j\in J$
    - $S_i$: supply at node $i\in I$
    - $D_j$: demand node at $j\in J$
- Decision variables:
    - $x_{ij}=$ the amount shipped from supply node $i\in I$ to demand node $j\in J$

<br>

$$
\begin{split}
\text{(NTP)}\quad & \text{minimize}\quad \sum\limits_{i\in I}\sum\limits_{j\in J}c_{ij}X_{ij}\\
&\begin{split}
\text{subject to}\quad\quad \sum\limits_{j\in J}&X_{ij}\le S_{i} &\forall i\in I\\
   \sum\limits_{i\in I}&X_{ij}\ge D_{j} \quad\quad&\forall j\in J\\
                       &X_{ij}\ge 0     &\forall i\in I, \forall j\in J\\ 
\end{split}
\end{split}
$$
