### (MCLP) Maximum Covering Location Problem
- Cover the most customer demands with no more than $p$ facilities.
- Sets:
    - $I$: set of customers
    - $J$: set of candidate facility locations
- Parameters:
    - $h_i$: demand at customer $i$ (annual)
    - $p$: number of facilities to locate
    - $a_{ij}=\begin{cases}1, \quad\text{if facitlity at } j\in J \text{ can cover customer } i\in I\\0, \quad\text{O.W.}\end{cases}$
- Decision variables:
    - $x_j=\begin{cases}1, \quad\text{if a facility is installed at } j\\0, \quad\text{O.W.}\end{cases}$
    - $z_{ij}=\begin{cases}1, \quad\text{if customer } i \text{ is covered by an open facility}\\0, \quad\text{O.W.}\end{cases}$
        
<br>

$$
\begin{split}
\text{(MCLP)}\quad & \text{maximize}\quad \sum\limits_{i\in I}h_{i}z_{i}\\
&\begin{split}
\text{subject to}\quad\quad &z_{i}\le\sum\limits_{j\in J}a_{ij}x_{j} & \forall i\in I\\
    \sum\limits_{j\in J}&x_{j}=p\\
    & x_{j}\in \{0, 1\}                & \forall j\in J\\
    & z_{i}\in \{0, 1\}\quad\quad\quad & \forall i\in I
\end{split}
\end{split}
$$
