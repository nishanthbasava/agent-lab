import numpy as np

GRID = [
    "S...#.",
    ".#....",
    ".#.T..",
    "...#..",
    ".#...G",
    "......",
]

class GridWorld:
    def __init__(self, grid, slip=0.1, step_cost=-0.04):
        self.grid = np.array([list(row) for row in grid])
        self.nrows, self.ncols = self.grid.shape
        self.slip = slip
        self.step_cost = step_cost
        self.n_states = self.nrows * self.ncols

    def _to_state(self, r, c): #this method converts the row and colummn to state, which is one number representing the agent's position
        return r * self.ncols + c

    def _to_rc(self, s):
        return divmod(s, self.ncols)
    
    MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    # if a = 0, agent moves up
    # if a = 1, agent moves right
    # if a = 2, agent moves down 
    # if a = 3, agent moves left

    def _move(self, s, a): #transition function (part of the environment's step function)
        r, c = self._to_rc(s)
        nr, nc = r + self.MOVES[a][0], c + self.MOVES[a][1]

        # off-grid or wall: stay put
        if not (0 <= nr < self.nrows and 0 <= nc < self.ncols) or self.grid[nr, nc] == "#":
            return s

        return self._to_state(nr, nc)

    def P(s, a):
        #Dynamics/Transition Probabilities table, will contain Transition Probabilities P(s' | s, a) and Expected Rewards (R(s, a, s'))
        # In Deep RL i typically wouldn't have such a clean and easy table, so for my next project, which will be an agent in CV, def check out different Deep RL options



        #handle terminal states later



        pass

    #next i gotta write DP



    #my env will expose this as P[s][a] -> [(prob, s_next, reward, done)]

        
        #this method will convert the agent's planned movement and current state into a new state
        #it will return the new state value, which should just be one integer. 

    #DP

    #Block 1 will have policy evaluation, value iteration, etc. But htis function consumes nothing except p(s', r | s, a) so i need a method for this.
    
    # DP methods never STEP the environment; they just do math over P. (sampling based methods in block 2 are ___)
