### (SCLP) Set-Covering Location Problem
- Service is provided to customers within distance $r$ of any facility.
- Sets:
    - $I$: set of customers
    - $J$: set of candidate facility locations
- Parameters:
    - $a_{ij}$: whether customer $i$ is covered by facility at $j$ (distance $\le r$)<br>
      $a_{ij}=\begin{cases}1, \quad\text{if customer } i \text{ is covered by a facility at } j\\0, \quad\text{O.W.}\end{cases}$
- Decision variables:
    - $x_j=\begin{cases}1, \quad\text{if a facility is installed at } j\\0, \quad\text{O.W.}\end{cases}$
        
<br>

$$
\begin{split}
\text{SCLP}\quad & \text{minimize}\quad \sum\limits_{j\in J}x_{j}\\
&\begin{split}
\text{subject to}\quad \sum\limits_{j\in J}&a_{ij}\ge1 & \forall i\in I\\
    & x_{j}\in \{0, 1\}\quad\quad & \forall j\in J\\
\end{split}
\end{split}
$$
