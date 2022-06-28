### (VRP) Vehicle Routing Problem
- Optimize a set of routes such that
    - all begin and end at the same location(s)
    - serve a set of customers
- Sets:
    - $N=\{1,...,n\}$: set of nodes
- Parameters:
    - $K$: number of vehicles
    - $c_{ij}$: distance (or transportation cost) from node $i$ to node $j$
- Decision variables:
    - $x_j=\begin{cases}1, \quad\text{if the route goes from }i\text{ to }j\\0, \quad\text{O.W.}\end{cases}$
- Notes:
    - $N^{-}=N\backslash\{0\}$, where $N=\{0\}$ is the start and end location
    - $v(S)$: minimum number of vehicles to visit all customers in subset $S$

<br>

$$
\begin{split}
\text{(VRP)}\quad & \text{minimize}\quad \sum\limits_{i,j\in N}c_{ij}x_{ij}\\
&\begin{split}
\text{subject to}\quad\quad \sum\limits_{i\in N}&x_{ih} +\sum\limits_{j\in N}x_{hj}=2 \quad\quad &\forall h\in N^{-}\\
    \sum\limits_{i\in N}&x_{0j}=2K\\
    \sum\limits_{i,j\in S}&x_{ij}\le |S|-v(S) &\forall S\subseteq N^{-}:S\ne \emptyset\\
    &x_{ij}\in \{0,1\}     &\forall i,j\in N^{-}\\ 
    &x_{0j}\in \{0,1,2\}   &\forall j\in N^{-}\\ 
\end{split}
\end{split}
$$
