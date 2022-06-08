### (NDP) Node Design Problem
- Minimize the total fixed charges and transportation cost.
- Sets:
    - $I$: set of customers locations
    - $J$: set of candidate facility locations
    - $K$: set of candidate distribution center locations
    - $L$: set of products
- Parameters:
    - $h_{il}$: (annual) demand of customer $i\in I$ for product $l\in L$
    - $v_j$: capacity of distribution center $j\in J$
    - $b_k$: capacity of plant $k\in K$
    - $s_l$: units of capacity of plant $\in K$
    - $f_j$: fixed (annual) cost to open a distribution center at site $l\in L$
    - $g_k$: fixed (annual) cost to open a plant at site $k\in K$
    - $c_{ijk}$: cost to transport 1 unit of product $l\in L$ from distribution center at $j$ to customer $i$
    - $d_{jkl}$: cost to transport 1 unit of product $l\in L$ from plant at $k$ to distribution center at $j$
- Decision variables:
    - $x_j=\begin{cases}1, \quad\text{if a distribution center is opened at } j\in J\\0, \quad\text{O.W.}\end{cases}$
    - $z_k=\begin{cases}1, \quad\text{if a plant is opened at } k\in K\\0, \quad\text{O.W.}\end{cases}$
    - $y_{ijl}=\text{the number of units of product }l\in L\text{ shipped from distribution center at }j\text{ to customer }i$
    - $w_{jkl}=\text{the number of units of product }l\in L\text{ shipped from plant at }k\text{ to distribution center at }j$

<br>

$$
\begin{split}
\text{(NDP)}\quad & \text{minimize}\quad \sum\limits_{j\in J}f_{j}x_{j} + \sum\limits_{k\in K}g_{k}z_{k} + \sum\limits_{l\in L}(\sum\limits_{j\in J}\sum\limits_{i\in I}c_{ijl}y_{ijl} + \sum\limits_{k\in K}\sum\limits_{j\in J}d_{jkl}w_{jkl})\\
&\begin{split}
\text{subject to}\quad\quad &\sum\limits_{j\in J}y_{ijl} = h_{il}                  &\forall i\in I, \forall l\in L\\
    &\sum\limits_{i\in I}\sum\limits_{l\in L}s_{l} y_{ijl} \le v_{j}x_{j}          &\forall j\in J\\
    &\sum\limits_{k\in K}w_{jkl}=\sum\limits_{i\in I}y_{ijl}                       &\forall j\in J, \forall l\in L\\
    &\sum\limits_{j\in J}\sum\limits_{l\in L}s_{l}w_{jkl} \le b_{k}z_{k}\quad\quad &\forall k\in K\\
    &x_{j}, z_{k}\in \{0, 1\} &\forall j\in J, \forall k\in K\\
    &y_{ijl}, w_{jkl}\ge 0    &\forall i\in I, \forall j\in J, \forall k\in K, \forall l\in L\\ 
\end{split}
\end{split}
$$
