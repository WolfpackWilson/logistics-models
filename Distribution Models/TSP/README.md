### (TSP) Traveling Salesman Problem
- Find the shortest route through $n$ nodes such that 
    - the route begins and ends at the same location 
    - the route visits every node
- Sets:
    - $N=\{1,...,n\}$: set of nodes
- Parameters:
    - $c_{ij}$: distance (or transportation cost) from node $i$ to node $j$
- Decision variables:
    - $x_j=\begin{cases}1, \quad\text{if the tour goes from }i\text{ to }j\\0, \quad\text{O.W.}\end{cases}$

<br>

$$
\begin{split}
\text{(TSP)}\quad & \text{minimize}\quad \sum\limits_{i,j\in N}c_{ij}x_{ij}\\
&\begin{split}
\text{subject to}\quad\quad \sum\limits_{i\in N}&x_{ih}\le +\sum\limits_{j\in N}x_{hj}\quad\quad &\forall h\in N\\
   \sum\limits_{i,j\in S}&x_{ij}\le |S|-1 &\forall S\subseteq N: 2\le |S|\le n-1\\
        &x_{ij}\in \{0,1\}     &\forall i,j\in N\\ 
\end{split}
\end{split}
$$
