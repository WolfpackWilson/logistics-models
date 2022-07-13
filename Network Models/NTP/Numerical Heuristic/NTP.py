# import libraries
import time
import random as r
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2

BLACK = (0, 0, 0)
CYAN = (0, 255, 255)

ORIGIN = (0, 0)
SIZE_X = 200
SIZE_Y = 150

INF = int(1e9)

# set seaborn settings
sns.reset_orig()
sns.set_style('darkgrid')
sns.set_palette('Set1')

# seed random
r.seed(int(time.time()))


class Cell:
    """An empty cell."""

    def __init__(self, size: tuple = (SIZE_X, SIZE_Y)):
        # numeric values
        self._value = None

        # image creation
        self.size = size

        s0, s1 = size
        self.img = np.ones((s1, s0, 3), dtype='uint8') * 255

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.add_text(self.size, str(value), 1.5, 2)

    def add_text(self, size: tuple, text: str, scale: float = 1, thickness: float = 1, offset: tuple = ORIGIN,
                 color: tuple = BLACK):
        """Center text and add to the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, scale, thickness)[0]

        text_x = (size[0] - (textsize[0])) // 2 + offset[0]
        text_y = (size[1] + (textsize[1])) // 2 + offset[1]

        self.img = cv2.putText(
            img=self.img,
            text=text,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=scale,
            color=color,
            thickness=thickness
        )

    def imshow(self):
        """Display the image of the cell."""
        cv2.imshow('Cell', self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()


class PCell(Cell):
    """A cell of the heuristic algorithm."""

    def __init__(self, cost: float, size: tuple = (SIZE_X, SIZE_Y)):
        super(PCell, self).__init__(size)

        # numeric values
        self.cost = cost

        # image creation
        self.size = size
        self.sym = None
        self.mk_img(size, cost)

    def mk_img(self, size: tuple, cost: float):
        """Create the image using OpenCV."""
        s0, s1 = size

        self.img = cv2.rectangle(self.img, ORIGIN, (s0 // 3, s1 // 3), BLACK, 4)
        self.img = cv2.rectangle(self.img, ORIGIN, (s0 + 5, s1 + 5), BLACK, 10)

        self.add_text((s0 // 3, s1 // 3), str(int(cost)), 1, 2)

    def imshow(self):
        """Display the image of the cell."""
        s0, s1 = self.size
        self.img = cv2.rectangle(self.img, ORIGIN, (s0, s1), BLACK, 10)

        cv2.imshow('PCell', self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def add_sym(self, sym: str):
        """Adds the symbol to the image."""
        self.sym = sym

        s0, s1 = self.size
        shape = (s0 // 3, s1 // 3)
        offset = (s0 - s0 // 3, s1 - s1 // 3)
        self.add_text(shape, sym, 1.5, 2, offset)


class Grid:
    """A grid of cells used to solve display the heuristic."""

    def __init__(self, costs: np.ndarray):
        cells = np.vectorize(PCell)(costs)

        self.cells = cells
        self.cell_size = cells[0, 0].size
        self.shape = costs.shape

        self.costs = costs
        self._lines = []
        self._circles = []

        self.img = None
        self.mk_img()

    def __getitem__(self, item):
        return self.cells[item]

    def mk_img(self):
        """Concatenate the cells to create an image of the grid."""
        rows = []
        for i in self.cells:
            row = [cell.img for cell in i]
            rows.append(np.concatenate(row, axis=1))

        c0, c1 = self.cell_size
        g0, g1 = self.shape
        size = (c0 * g1, c1 * g0)

        img = np.concatenate(rows)
        self.img = cv2.rectangle(img, ORIGIN, size, BLACK, 10)

        for circle in self._circles:
            self._add_circle(**circle)

        for line in self._lines:
            self._add_line(**line)

    def reset(self):
        """Reset the grid image (removes circles and lines)."""
        self._circles = []
        self._lines = []
        self.mk_img()

    def imshow(self):
        """Display the image of the grid."""
        cv2.imshow('Grid', self.img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def add_line(self, cell1: tuple, cell2: tuple):
        """Add a line between two symbol areas."""
        self._lines.append(dict(cell1=cell1, cell2=cell2))
        self._add_line(cell1, cell2)

    def _add_line(self, cell1: tuple, cell2: tuple):
        """Add a line between two symbol areas."""
        s0, s1 = self.cell_size
        center_offset = (s0 - s0 // 6, s1 - s1 // 6)

        def point(cell): return [
            cell[1] * s0 + center_offset[0],
            cell[0] * s1 + center_offset[1]
        ]

        p1 = point(cell1)
        p2 = point(cell2)

        axis = cell1[1] == cell2[1]
        forward = cell1[not axis] < cell2[not axis]
        offset = self.cell_size[axis] // 6 * (-1 if forward else 1)

        p1[axis] -= offset
        p2[axis] += offset

        self.img = cv2.line(
            self.img, p1, p2,
            color=CYAN,
            thickness=2
        )

    def add_circle(self, cell: tuple):
        """Add a circle around the center of a cell."""
        self._circles.append(dict(cell=cell))
        self._add_circle(cell)

    def _add_circle(self, cell: tuple):
        """Add a circle around the center of a cell."""
        s0, s1 = self.cell_size
        offset = (s0 // 2, s1 // 2)

        center = (cell[1] * s0 + offset[0], cell[0] * s1 + offset[1])
        radius = int(s1 / 2.5)
        self.img = cv2.circle(self.img, center, radius, BLACK, 2)


class NTP:
    def __init__(self, costs, supplies, demands):
        self.supplies = supplies.copy()
        self.demands = demands.copy()

        self.costs = costs.copy()
        self.grid = Grid(costs)

        self.obj_value = INF
        self.iter = 0

        self.img = None
        self.mk_img(supplies, demands, 'Supply', 'Demand')
        self.steps = [{
            'img': self.img,
            'text': 'Initialize'
        }]

        self.solution = {'solved': False}

    def mk_img(self, right, bottom, rheader, bheader):
        c0, c1 = self.grid.cell_size
        g0, g1 = self.grid.shape

        right = [rheader] + list(right)
        bottom = [bheader] + list(bottom)

        # create a canvas from several empty cells
        size = (g0 + 2, g1 + 2)
        canvas = np.zeros(size, dtype=object)
        for idx, _ in np.ndenumerate(canvas):
            canvas[idx] = Cell((c0, c1))

        # add text to the bottom side
        for j in range(size[1] - 1):
            canvas[g0 + 1, j].add_text(
                (c0, c1),
                str(bottom[j]),
                1.5, 2
            )

        # add text to the right side
        for i in range(size[0] - 1):
            canvas[i, g1 + 1].add_text(
                (c0, c1),
                str(right[i]),
                1.5, 2
            )

        # concatenate cells into a grid
        rows = []
        for i in canvas:
            row = [cell.img for cell in i]
            rows.append(np.concatenate(row, axis=1))
        canvas = np.concatenate(rows)

        self.grid.mk_img()
        canvas[
        c1:c1 + c1 * self.grid.shape[0],
        c0:c0 + c0 * self.grid.shape[1]
        ] = self.grid.img

        # add the objective value
        if self.obj_value < INF:
            s0, s1 = self.grid.cell_size

            canvas = cv2.putText(
                img=canvas,
                text=f'Objective value = {self.obj_value}',
                org=(s0 // 2, s1 // 2),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=BLACK,
                thickness=2
            )

        self.img = canvas

    def imshow(self, img=None):
        """Display the image of the grid."""
        cv2.imshow('NTP', self.img if img is None else img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def _save_step(self, right, bottom, rheader, bheader, text):
        """Update the image and save the step."""
        self.mk_img(right, bottom, rheader, bheader)
        self.steps.append({
            'img': self.img,
            'text': text
        })

    def solve(self):
        self._nw_corner()
        while self._solve_step():
            self.iter += 1

        print(f'Terminated in {self.iter} iterations')
        self.solution.update(dict(
            iter=self.iter,
            steps=self.steps
        ))

    def _nw_corner(self):
        """Perform the northwest corner method to begin the heuristic."""
        supplies = self.supplies
        demands = self.demands

        row = 0
        for col in range(self.grid.shape[1]):
            while demands[col] > 0:
                if supplies[row] >= demands[col]:
                    self.grid[row, col].value = demands[col]
                    supplies[row] -= demands[col]
                    demands[col] = 0
                else:
                    self.grid[row, col].value = supplies[row]
                    demands[col] -= supplies[row]
                    supplies[row] = 0
                    row += 1

        self._save_step(supplies, demands, 'Supply', 'Demand', 'Northwest method')

    def _solve_step(self):
        """Populate the grid with values"""
        basic = []
        slack = []

        # track basic and slack variables
        for idx, val in np.ndenumerate(self.grid.cells):
            if not val.value:
                slack.append(idx)
            else:
                basic.append(idx)

        self.solution.update(dict(
            cells=basic,
            values=[self.grid[cell].value for cell in basic],
            costs=[self.costs[cell] for cell in basic])
        )

        # find objective value and terminate if it is not improving
        old_value = self.obj_value
        self.obj_value = sum(self.grid[cell].cost * self.grid[cell].value for cell in basic)
        self.solution['obj_value'] = self.obj_value

        if self.obj_value >= old_value:
            self.mk_img(
                ['' for _ in range(len(self.supplies))],
                ['' for _ in range(len(self.demands))],
                '',
                '',
            )
            self.steps[-1]['img'] = self.img

            error = 'Infeasible using this heuristic.'
            print(error)
            self.solution.update(dict(error=error))

            return 0

        # calculate Vs and Ws
        v, w = self._calc_vw(basic)
        self.solution['v'] = v
        self.solution['w'] = w

        if len(v) == 0:
            error = 'Cannot calculate V and W.'
            print(error)
            self.solution.update(dict(error=error))

            return 0
        self._save_step(v, w, 'Vi', 'Wj', 'Solve for Vs and Ws')

        # calculate slack values and circle them
        min_slack, min_value = self._calc_slack(slack, v, w)
        self._save_step(v, w, 'Vi', 'Wj', 'Calculate the slack values and find the lowest slack variable.')

        # exit if all slack variables are positive
        if min_value > 0:
            print(f'Objective value = {self.obj_value}')
            self.solution['solved'] = True

            return 0

        # find and add path to image
        path = self._add_path(min_slack, basic)
        self._save_step(v, w, 'Vi', 'Wj', 'Find the optimal path starting from the lowest slack variable.')

        # redistribute values along the path
        self._update_grid(path, basic)
        self._save_step(
            ['' for _ in range(len(v))],
            ['' for _ in range(len(w))],
            '',
            '',
            'Find the smallest basic variable and then add and subtract it across the path.'
        )

        return 1

    def _calc_vw(self, basic):
        """Calculate Vs and Ws"""
        # initialize v and w
        v = np.ones(self.grid.shape[0], dtype=int) * INF
        v[0] = 0
        w = np.ones(self.grid.shape[1], dtype=int) * INF

        def is_stuck(list_):
            """Check if the calculation is stuck in a loop."""
            return all(v[i] == w[j] == INF for i, j in list_)

        # begin calculations
        stack = basic.copy()
        while stack:
            if is_stuck(stack):
                return [], []

            i, j = stack.pop(0)
            if v[i] == w[j] == INF:
                stack.append((i, j))
            elif v[i] == INF:
                v[i] = w[j] - self.costs[i, j]
            elif w[j] == INF:
                w[j] = self.costs[i, j] + v[i]

        return v, w

    def _calc_slack(self, slack, v, w):
        """Calculate slack values and circle them."""
        min_slack = slack[0]
        min_value = INF

        for cell in slack:
            i, j = cell
            value = self.costs[cell] + v[i] - w[j]

            self.grid[cell].value = value
            self.grid.add_circle(cell)

            if value < min_value:
                min_value = value
                min_slack = cell

        return min_slack, min_value

    def _add_path(self, min_slack, basic):
        """Find the optimal path starting from the lowest slack variable and add it to the image."""
        start = min_slack
        path = self._find_path([start] + basic, [start])

        for p1, p2 in zip(path[:-1], path[1:]):
            self.grid.add_line(p1, p2)

        for i in range(len(path) - 1):
            cell = path[i]
            self.grid[cell].add_sym('-' if i % 2 else '+')

        return path

    def _find_path(self, cells, path, axis=0):
        """Recursively search for a valid path."""
        start = path[0]
        curr = path[-1]

        for cell in cells:
            if cell[axis] == curr[axis]:
                if cell == start and curr != start:
                    return path + [start]
                elif curr != cell:
                    result = self._find_path(cells, path + [cell], 0 if axis else 1)
                    if result and result[0] == result[-1]:
                        return result

    def _update_grid(self, path, basic):
        """Find the smallest basic variable and then add and subtract it across the path."""
        min_value = INF
        for i in range(len(path) - 1):
            value = self.grid[path[i]].value

            if i % 2 and value < min_value:
                min_value = value

        grid = Grid(self.costs)
        self.grid.reset()

        for cell in basic:
            if cell not in path:
                grid[cell].value = self.grid[cell].value

        for i in range(len(path))[1:-1]:
            cell = path[i]
            value = self.grid[cell].value + (-1 if i % 2 else 1) * min_value

            if value > 0:
                grid[cell].value = value

        grid[path[0]].value = min_value

        self.grid = grid
        self.grid.mk_img()

    def plot(self, i_coord, j_coord):
        """Plot the NTP solution."""
        if not self.solution['solved']:
            print('Invalid solution.')
            return

        def mean(*arg):
            return sum(*arg) / len(*arg)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x='X', y='Y', data=j_coord, s=50, legend=False, label="Demand")
        sns.scatterplot(x='X', y='Y', data=i_coord, s=150, color='black', marker='^', label="Supply")

        for path, nb in zip(self.solution['cells'], self.solution['values']):
            point1 = [i_coord['X'][path[0]], i_coord['Y'][path[0]]]
            point2 = [j_coord['X'][path[1]], j_coord['Y'][path[1]]]
            x_values = [point1[0], point2[0]]
            y_values = [point1[1], point2[1]]

            plt.plot(x_values, y_values, 'r', linestyle=":", zorder=0)
            plt.text(
                mean(x_values), mean(y_values), str(nb), backgroundcolor='#eaeaf2',
                horizontalalignment='center', verticalalignment='center'
            )

        plt.title('Demand Coverage')
        plt.show()


def gen_data(valid_problem: bool = True, **kwargs):
    """Generate a valid set of data for the NTP heuristic."""
    while True:
        i_data, j_data, dist_ij = _gen_data(**kwargs)
        S = i_data['Supply'].values
        D = j_data['Demand'].values
        costs = dist_ij

        p = NTP(costs, S, D)
        p.solve()

        if not valid_problem or p.solution['solved']:
            return i_data, j_data, dist_ij


def _gen_data(size: tuple, supply_min: int, supply_max: int, coord_bound: int = 10):
    """Generate a set of data for the NTP heuristic."""
    i_len, j_len = size

    def rcoord(nb): return r.randrange(-nb, nb + 1)
    def eucl(x1, x2, y1, y2): return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    # add supply
    i_data = pd.DataFrame()
    i_data['X'] = [rcoord(coord_bound) for _ in range(i_len)]
    i_data['Y'] = [rcoord(coord_bound) for _ in range(i_len)]

    supply = [r.randrange(supply_min, supply_max) for _ in range(i_len)]
    i_data['Supply'] = supply

    # add demand
    j_data = pd.DataFrame()
    j_data['X'] = [rcoord(coord_bound) for _ in range(j_len)]
    j_data['Y'] = [rcoord(coord_bound) for _ in range(j_len)]

    # find valid demand
    min_ = min(supply) * i_len // j_len
    max_ = max(supply) * i_len // j_len
    demand = [r.randrange(min_, max_) for _ in range(j_len)]

    while sum(demand) > sum(supply):
        demand = [r.randrange(min_, max_) for _ in range(j_len)]
        min_ -= 1 if min_ > 1 else 0
        max_ -= 1

    j_data['Demand'] = demand

    dist_ij = np.array([
        [
            int(eucl(i[0], i[1], j[0], j[1]))
            for j in j_data[['X', 'Y']].values
        ] for i in i_data[['X', 'Y']].values
    ])

    return i_data, j_data, dist_ij


if __name__ == '__main__':
    # import data
    i_data, j_data, dist_ij = gen_data(
        size=(3, 4),
        supply_min=80,
        supply_max=150,
        coord_bound=10
    )

    # declare the parameters and sets
    S = i_data['Supply'].values
    D = j_data['Demand'].values

    cost = 1
    costs = dist_ij * cost

    p = NTP(costs, S, D)
    p.solve()

    for step in p.steps:
        p.imshow(step['img'])

    p.plot(i_data, j_data)

    print('Done')

