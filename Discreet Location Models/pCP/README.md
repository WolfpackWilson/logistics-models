### ($p$CP) $p$-Center Problem
- Locate $p$ facilities (centers) to minimize the maximum distance between each customer and their nearest (assigned) facility (or center).
- Sets:
    - $I$: set of customers
    - $J$: set of candidate facility locations
- Parameters:
    - $c_{ij}$: cost to travel between customer $i$ and candidate facility at $j$
    - $p$: number of facilities to locate
- Decision variables:
    - $x_j=\begin{cases}1, \quad\text{if a facility is installed at } j\\0, \quad\text{O.W.}\end{cases}$
    - $y_{ij}=\begin{cases}1, \quad\text{if customer } i \text{ is assigned to a facility at } j\\0, \quad\text{O.W.}\end{cases}$
    - $r=\text{maximum distance, over all } i\in I\text{, from }i\text{ to its assigned facility.}$
        
<br>

$$
\begin{split}
\text{(}p\text{CP)}\quad & \text{minimize}\quad\quad r\\
&\begin{split}
\text{subject to}\quad\quad \sum\limits_{j\in J}&y_{ij}=1 & \forall i\in I\\
    & y_{ij}\le x_{j}                     & \forall i\in I, \forall j\in J\\
    \sum\limits_{j\in J}&x_{j}=p\\
    \sum\limits_{j\in J}&c_{ij}y_{ij}\le r & \forall i\in I\\
    & x_{j}\in \{0, 1\}                   & \forall j\in J\\
    & y_{ij}\in \{0, 1\}\quad\quad        & \forall i\in I, \forall j\in J\\
\end{split}
\end{split}
$$
