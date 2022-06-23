### (SPP) Shortest Path Problem
- Find the shortest path from a specified node $s\in N$ to another node $t\in N$
- Sets:
    - $N$: set of nodes
    - $A$: set of arcs
- Parameters:
    - $c_{ij}$: cost of traversing arc $(i,j)\in A$
- Decision variables:
    - $X_{(i,j)}=1$, if arc $(i,j)$ is on the shortest path from node $s\in N$ to node $t\in N$; or 0, O.W.

<br>

$$
\begin{split}
\text{(SPP)}\quad & \text{minimize}\quad \sum\limits_{i\in I}\sum\limits_{j\in J}c_{ij}X_{ij}\\
&\begin{split}
\text{subject to}\quad\quad \sum\limits_{j\in B_i}&X_{(j,i)}-\sum\limits_{j\in A_i}X_{(i,k)}=\begin{cases}-1 \text{ }&\text{if node }i=s\\0 &i\in N,i\ne s,i\ne t\\1 &\text{if node }i=t\end{cases} \quad\quad&\forall i\in N\\
        &X_{ij}\in \{0,1\} \quad\quad\forall (i,j)\in A\\ 
\end{split}
\end{split}
$$
