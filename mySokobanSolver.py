'''

The partially defined functions and classes of this module 
will be called by a marker script. 

You should complete the functions and classes according to their specified interfaces.
 

'''

import search

import sokoban
import itertools


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9370331, 'Jiaming', 'Chen'), (8920281, 'Dongmin', 'Park'), (9713581,'Christopher', 'Ayling' ) ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def taboo_cells(warehouse):
    '''  
    Identify the taboo cells of a warehouse. A cell is called 'taboo' 
    if whenever a box get pushed on such a cell then the puzzle becomes unsolvable.  
    When determining the taboo cells, you must ignore all the existing boxes, 
    simply consider the walls and the target  cells.  
    Use only the following two rules to determine the taboo cells;
     Rule 1: if a cell is a corner and not a target, then it is a taboo cell.
     Rule 2: all the cells between two corners along a wall are taboo if none of 
             these cells is a target.
    
    @param warehouse: a Warehouse object

    @return
       A string representing the puzzle with only the wall cells marked with 
       an '#' and the taboo cells marked with an 'X'.  
       The returned string should NOT have marks for the worker, the targets,
       and the boxes.  
    '''
    ##         "INSERT YOUR CODE HERE"    
    deadlock_cells = get_all_deadlock_cells(warehouse)
    
    
    # Create a string to represent the marked puzzle
    # Create a matrix, and mark by 'X' and '#'.
    X,Y = zip(*warehouse.walls)
    x_size, y_size = 1+max(X), 1+max(Y)    
    vis = [[" "] * x_size for y in range(y_size)]
    for (x,y) in warehouse.walls:
        vis[y][x] = "#"
    for cell in deadlock_cells:
        vis[cell[1]][cell[0]] = "X"
        
    return "\n".join(["".join(line) for line in vis])

def get_all_deadlock_cells(warehouse):
    '''
    The function is to find all of deadlocks
    
    @param warehouse: a Warehouse object
    
    @return:
        return a set of deadlocks
    '''
    valid_cells = get_valid_cells(warehouse)
    return mark_deadlock_cells(warehouse, valid_cells)


def mark_deadlock_cells(warehouse, valid_cell):
    '''
    The function is to find all of deadlock cells in the puzzle
    
    @param warehouse: a Warehouse object
           valid_cell: a set of valid cells where the worker can go

    @return
        a set of deadlock will be returned
    '''
    #Remove all of target cells, which do need to be consider deadlock case
    for target_cell in warehouse.targets:
        valid_cell.discard(target_cell)
    
    #Get the corner deadlock
    deadlocks = set([cell for cell in valid_cell
        if((get_neighbour_cells(cell)['top'] in warehouse.walls or 
            get_neighbour_cells(cell)['bottom'] in warehouse.walls) and
            (get_neighbour_cells(cell)['left'] in warehouse.walls or 
             get_neighbour_cells(cell)['right'] in warehouse.walls))])
    
    # Get the deadlock along the walls
    deadlock_alongWall = set()
    for cell1, cell2 in itertools.combinations(deadlocks, 2): 
        x1, y1 = cell1[0], cell1[1]
        x2, y2 = cell2[0], cell2[1]
        if x1 == x2:
            if y1>y2:
                y1, y2 = y2, y1
            ## check whether there is a target or wall between them
            TargetOrWallsBetweenThem = False
            for y in range(y1+1,y2):
                if (x1,y) in warehouse.targets or (x1,y) in warehouse.walls:
                    TargetOrWallsBetweenThem = True
                    break
            if TargetOrWallsBetweenThem:
                continue
            
            ##check whether they are along the wall 
            alongWall_left = not False in [False for y in range(y1, y2+1) if (x1-1, y) not in warehouse.walls]
            alongWall_right = not False in [False for y in range(y1, y2+1) if (x1+1, y) not in warehouse.walls]

            # append all deadlock cells along the wall into set
            if alongWall_left or alongWall_right:
                deadlock_alongWall |=  set([(x1, y) for y in range(y1+1, y2)])
        
        if y1 == y2:
            if x1 > x2:
                x1, x2 = x2, x1
            ## check whether there is target between them
            TargetOrWallsBetweenThem = False
            for x in range(x1+1,x2):
                if (x,y1) in warehouse.targets or (x,y1) in warehouse.walls:
                    TargetOrWallsBetweenThem = True
                    break
            if TargetOrWallsBetweenThem:
                continue
            
            ##check whether they are along the wall                            
            alongWall_top = not False in [False for x in range(x1, x2+1) if (x+1, y1-1) not in warehouse.walls]
            alongWall_bottom = not False in [False for x in range(x1, x2+1) if (x, y1+1) not in warehouse.walls]
            # append all deadlock cells along the wall into set
            if alongWall_top or alongWall_bottom:
                deadlock_alongWall |= set([(x,y1) for x in range(x1+1, x2)])      
    
    # Merge all of deadlock to the single set
    deadlocks |= deadlock_alongWall

    return deadlocks


def get_valid_cells(warehouse):
    '''
    The function is to find all of valid cell within the walls and only consider
    about # notation.
    
    @param warehouse: a Warehouse object
    
    @return 
        a set of valid cells
    '''
    frontier = set()
    explored = set()
    frontier.add(warehouse.worker)

    while frontier:
        curr_point = frontier.pop()
        explored.add(curr_point)

        neighbour_cells = get_neighbour_cells(curr_point)
       
        for neighbour_cell in neighbour_cells.values():
            if (neighbour_cell not in frontier and neighbour_cell not in explored
                and neighbour_cell not in warehouse.walls):
                frontier.add(neighbour_cell)
    return explored
        
def get_neighbour_cells(curr_point):
    '''
    This function is to find all of neighbour cells
    
    @param curr_point: the location of the point
    
    @return:
        dictionary of four neighbour cells
    '''
    
    curr_x, curr_y = curr_point
    return {'right':(curr_x+1, curr_y),'left':(curr_x-1, curr_y),
                'top':(curr_x, curr_y+1),'bottom':(curr_x, curr_y-1)}


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


class SokobanPuzzle(search.Problem):
    '''
    Class to represent a Sokoban puzzle.
    Your implementation should be compatible with the
    search functions of the provided module 'search.py'.
    
    	Use the sliding puzzle and the pancake puzzle for inspiration!
    
    '''
    ##         "INSERT YOUR CODE HERE"
    
    def __init__(self, warehouse, goal = None):
        self.initial = warehouse

        if goal is None:
            self.goal = warehouse.copy()
            self.goal.boxes = self.goal.targets
            self.alt_goal = False
        else:
            self.goal = warehouse.copy()
            self.goal.worker = goal
            self.alt_goal = True
        
        self.dead_locks = get_all_deadlock_cells(warehouse)
        
        self.original_boxes = []
        self.original_worker = self.goal.worker
        
    def actions(self, state):
        """
        Return the list of actions that can be executed in the given state 
        if these actions do not push a box in a taboo cell.
        The actions must belong to the list ['Left', 'Down', 'Right', 'Up']        
        """
        actions = ['Left', 'Down', 'Right', 'Up']
        valid_actions = []
        
        self.original_boxes = state.boxes.copy()
        self.original_worker = state.worker

        for action in actions:
            temp_warehouse = check_each_action_and_move(state.copy(worker=self.original_worker, boxes=state.boxes.copy()), [action])
            
            if type(temp_warehouse) != str:
                #temp_warehouse != 'Failure'
                #the action not fail
                if set(temp_warehouse.boxes) & self.dead_locks == set():
                    # no box was pushed onto taboo cells
                    valid_actions.append(action)
                    
        return valid_actions
        
    
    def result(self, state, action):
        original_boxes = self.original_boxes.copy()
        return check_each_action_and_move(state.copy(worker=self.original_worker, boxes=state.boxes.copy()),[action])
    
    def print_solution(self, goal_node):
        if goal_node == None:
            return "The puzzle has no solution... It cannot be solved!!"
        
        # path is list of nodes from initial state (root of the tree)
        # to the goal_node
        path = goal_node.path()
        # print the solution
        print ("Solution takes {0} steps from the initial state".format(len(path)-1))
        print (path[0].state)
        print ("to the goal state")
        print (path[-1].state)
        print ("\nBelow is the sequence of moves\n")
        
        for node in path:
            print(node.action)
            print(node.state)
        
        return goal_node.solution()
    

    def return_solution(self, goal_node):
        if goal_node == None:
            return ['Impossible']
        
        path = goal_node.path()
        solution = []
        for node in path:
            solution.append(node.action)

        solution.remove(None)
        return solution


    def goal_test(self, state):
        if self.alt_goal:
            return (self.goal == state)
        else:
            return set(self.goal.boxes) == set(state.boxes)
        
    def path_cost(self, c, state1, action, state2):
        return c + 1
    
    def h(self, n):
        if self.alt_goal:
            #This obj is for worker moving
            return mDist(n.state.worker, self.goal.worker)
        else:
            #This obj is for boxes moving
            #need use some help function to get "which box should goto which specific target"
            #then use manhattan distance to calculate
            pass
            return 0
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_action_seq(warehouse, action_seq):
    '''
    
    Determine if the sequence of actions listed in 'action_seq' is legal or not.
    
    Important notes:
      - a legal sequence of actions does not necessarily solve the puzzle.
      - an action is legal even if it pushes a box onto a taboo cell.
        
    @param warehouse: a valid Warehouse object

    @param action_seq: a sequence of legal actions.
           For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
           
    @return
        The string 'Failure', if one of the action was not successul.
           For example, if the agent tries to push two boxes at the same time,
                        or push one box into a wall.
        Otherwise, if all actions were successful, return                 
               A string representing the state of the puzzle after applying
               the sequence of actions.  This must be the same string as the
               string returned by the method  Warehouse.__str__()
    '''
    
    ##         "INSERT YOUR CODE HERE"

    warehouse = check_each_action_and_move(warehouse, action_seq)
    
    return warehouse.__str__() if type(warehouse)!=str else warehouse

    
def check_each_action_and_move(warehouse, action_seq):
    '''
    Same purpose as check_action_seq function
    NB: It does not check if it pushes a box onto a taboo cell.
    
    @param warehouse: a Warehouse object
    @param action_seq: a list of actions
    
    @return
        a altered warehouse
    '''

    for action in action_seq:
        worker_x, worker_y = warehouse.worker
        
        #Get localtion of two cells we should check
        if action == 'Left':
            cell1 = (worker_x-1, worker_y)
            cell2 = (worker_x-2, worker_y)
            
        elif action == 'Right':
            cell1 = (worker_x+1, worker_y)
            cell2 = (worker_x+2, worker_y)
        
        elif action == 'Up':
            cell1 = (worker_x, worker_y-1)
            cell2 = (worker_x, worker_y-2)
        
        elif action == 'Down':
            cell1 = (worker_x, worker_y+1)
            cell2 = (worker_x, worker_y+2)            
        
        #Check whether the worker push walls
        if cell1 in warehouse.walls:
            return 'Failure'

        if cell1 in warehouse.boxes:
            if cell2 in warehouse.boxes or cell2 in warehouse.walls:
                #push two boxes or the box has already nearby the wall, faliure
                return 'Failure'
            #Only push one box
            warehouse.boxes.remove(cell1)
            warehouse.boxes.append(cell2)

        warehouse.worker = cell1 

    return warehouse


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_elem(warehouse):
    '''    
    This function should solve using elementary actions 
    the puzzle defined in a file.
    
    @param warehouse: a valid Warehouse object

    @return
        A list of strings.
        If puzzle cannot be solved return ['Impossible']
        If a solution was found, return a list of elementary actions that solves
            the given puzzle coded with 'Left', 'Right', 'Up', 'Down'
            For example, ['Left', 'Down', Down','Right', 'Up', 'Down']
            If the puzzle is already in a goal state, simply return []
    '''
    
    ##         "INSERT YOUR CODE HERE"
    elementary_actions = []

    ##Load macro actions to execute from solve_sokoban_macro
    ##Currently using test data
    macro_actions = solve_sokoban_macro(warehouse)
#    macro_actions = [((3, 6), 'Down'), ((3, 5), 'Down'), ((3, 4), 'Down')]

    print("Initial State:")
    print(warehouse)

    for macro_action in macro_actions: #format is ((r, c), 'Direction')
        #macro action is in ((r, c), 'Direction')
        #warehouse.objects are in (x, y)
        #Calculate the position the worker must be in to move the box
        move_to = (macro_action[0][1], macro_action[0][0])
        
        if macro_action[1] == 'Left':
            move_to = (move_to[0]+1, move_to[1])
        elif macro_action[1] == 'Right':
            move_to = (move_to[0]-1, move_to[1])
        elif macro_action[1] == 'Up':
            move_to = (move_to[0], move_to[1]+1)
        elif macro_action[1] == 'Down':
            move_to = (move_to[0], move_to[1]-1)
            
        #Create SokobanPuzzle object and set goal
        sp = SokobanPuzzle(warehouse, goal = move_to)
        if warehouse.worker == move_to:
            #move worker in desired direction
            warehouse = check_each_action_and_move(warehouse, [macro_action[1]])
        else:
            #move worker to desired location
            sol = search.astar_graph_search(sp)    

            #move worker in desired direction
            warehouse = check_each_action_and_move(sol.state, [macro_action[1]])
            #update list of required elemtary actions
            elementary_actions.extend(sp.return_solution(sol))

        #update list of required elemtary actions
        elementary_actions.append(macro_action[1])

        print("\nafter:" + str(macro_action))
        print(warehouse)
        print("\nElementary Actions Executed:")
        print(elementary_actions)


    print("\nFinal State:")
    print(warehouse)
    print("\nElementary Actions Required:")
    print(elementary_actions)
    
    return elementary_actions
    

def mDist(pos1, pos2):
    """
    Finds the manhattan distance between two points
    
    @param pos1: location (tuple)
    @param pos2: location (tuple)
    
    @return
        Manhattan distance of two locations
    """
    return abs((pos1[0] - pos2[0])) + abs((pos1[1] - pos2[1]))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def can_go_there(warehouse, dst):
    '''    
    Determine whether the worker can walk to the cell dst=(row,col) 
    without pushing any box.
    
    @param warehouse: a valid Warehouse object

    @return
      True if the worker can walk to cell dst=(row,col) without pushing any box
      False otherwise
    '''
    
    ##         "INSERT YOUR CODE HERE"
    frontier = set()
    explored = set()
    frontier.add(warehouse.worker)

    while frontier:
        curr_point = frontier.pop()
        if curr_point == (dst[1],dst[0]):
            return True
        explored.add(curr_point)

        neighbour_cells = get_neighbour_cells(curr_point)
       
        for neighbour_cell in neighbour_cells.values():
            if (neighbour_cell not in frontier and 
                neighbour_cell not in explored and
                neighbour_cell not in warehouse.walls and 
                neighbour_cell not in warehouse.boxes):
                frontier.add(neighbour_cell)
   
    return False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_sokoban_macro(warehouse):
    '''    
    Solve using macro actions the puzzle defined in the warehouse passed as
    a parameter. A sequence of macro actions should be 
    represented by a list M of the form
            [ ((r1,c1), a1), ((r2,c2), a2), ..., ((rn,cn), an) ]
    For example M = [ ((3,4),'Left') , ((5,2),'Up'), ((12,4),'Down') ] 
    means that the worker first goes the box at row 3 and column 4 and pushes it left,
    then goes the box at row 5 and column 2 and pushes it up, and finally
    goes the box at row 12 and column 4 and pushes it down.
    
    @param warehouse: a valid Warehouse object

    @return
        If puzzle cannot be solved return ['Impossible']
        Otherwise return M a sequence of macro actions that solves the puzzle.
        If the puzzle is already in a goal state, simply return []
    '''
        
    if warehouse.targets == warehouse.boxes:
        return []
    
    sokoban_macro = SokobanMacro(warehouse)
    
    sol = search.astar_graph_search(sokoban_macro)
    
    macro_xy = sokoban_macro.return_solution(sol)
    
    #convert (x,y) to (r,c)
    macro_rc = []
    for action in macro_xy:
        macro_rc.append(((action[0][1], action[0][0]), action[1]))
    
    return macro_rc
    
    
class SokobanMacro(search.Problem):
    '''
    This class is for checking, moving box and returning the macro actions
    During searching, worker location and deadlock are always freshed
    '''
    def __init__(self, initial):
        '''
        Set initial state and goal state
        '''
        self.initial = initial
        self.goal = initial.copy(boxes=initial.targets)
    
        self.current_boxes = []
    
    def actions(self, state):
        '''
        The function is to find all pushable boxes in current situation
        
        @param state: a Warehouse object
        
        @return
            The locations of boxes which can be push by current state of worker, 
            and also return direction
        '''
        
        possible_actions = []
        #backup current boxes location
        self.current_boxes = state.boxes.copy()
        dead_locks = get_all_deadlock_cells(state)
        
        #get all of pushable boxes with direction, and worker locations which nearby them
        possible_boxes_pushed, worker_locations = self.boxes_can_be_pushed_with_worker_location(state.copy())
        
        #check if each action is valid or not
        for box in possible_boxes_pushed:
            #get the worker location which around boxes
            worker_around_one_box = set(get_neighbour_cells(box).values()) & set(worker_locations)
            
            for worker in worker_around_one_box:
                offset = worker[0]-box[0], worker[1]-box[1]
                
                #get the second cell to check if the boxes should be pushed or not
                second_cell = box[0]-offset[0], box[1]-offset[1]
                if second_cell not in dead_locks and second_cell not in state.boxes and second_cell not in state.walls:
                    #the second cell is not deadlocks/boxes/walls, so it can be pushed
                    if offset == (-1,0):
                        possible_actions.append((box, "Right"))
                    elif offset == (1,0):
                        possible_actions.append((box, "Left"))
                    elif offset == (0,-1):
                        possible_actions.append((box, "Down"))
                    elif offset == (0,1):
                        possible_actions.append((box, "Up"))
                    
        return possible_actions
        
    
    def result(self, state, action):
        '''
        The function is to move the boxes really.
        
        @param state: a Warehouse object
        @param action: a tuple of box location and direction
        
        @return
            a warehouse object which boxes was moved
        '''
        state = state.copy(boxes=self.current_boxes.copy())
        
        box_previous_location = action[0]
        
        state.boxes.remove(box_previous_location)
        state.worker = box_previous_location
        
        # for matching get_neighbour_cells result
        if action[1] == "Right":
            pos = "right"
        elif action[1] == "Left":
            pos = "left"
        elif action[1] == "Up":
            pos = "bottom"
        elif action[1] == "Down":
            pos = "top"
        
        #move box to new location
        state.boxes.append(get_neighbour_cells(box_previous_location)[pos])

        return state
    
    def goal_test(self, state):
        return set(state.boxes) == set(self.goal.boxes)
    
    def path_cost(self, c, state1, action, state2):
        return c+1
    
#    def h(self, state):
#        dist = 0       
#        for i in range(len(state.state.targets)):
#            num_boxes_not_on_targets = len(state.state.targets)-len(set(state.state.targets)&set(state.state.boxes))
#            if state.state.targets[i] not in state.state.boxes:
#                for index in range(len(state.state.boxes)):
#                    if state.state.boxes[index] not in state.state.targets:
#                        dist += mDist(state.state.boxes[index], state.state.targets[i])/num_boxes_not_on_targets
#        return dist
    
    def h(self, n):
    	"""
    	Returns the heuristic of node n.
    	Heuristic is the manhattan distance from all boxes
    	to their closest target.
    	"""
    	heur = 0
    	for box in n.state.boxes:
    		#Find closest target
    		closest_target = n.state.targets[0]
    		for target in n.state.targets:
    			if (mDist(target, box) < mDist(closest_target, box)):
    				closest_target = target
    				
    		#Update Heuristic
    		heur = heur + mDist(closest_target, box)              
    
    	return heur
    
    def boxes_can_be_pushed_with_worker_location(self, warehouse):
        '''
        The function is to find all the boxes the worker can push currently
        
        @param warehouse: a Warehouse object
        
        @return a tuple of boxes location set and worker location set
        '''
        dead_locks = get_all_deadlock_cells(warehouse)
        valid_cells = get_valid_cells(warehouse)
        
        boxes_can_be_pushed = set()
        worker_locations_nearby_boxes = set()
        #check all of cells worker can reach
        for valid_cell in valid_cells:
            #check any box can move to neighbour cell
            temp_boxes_can_be_pushed = set(warehouse.boxes) & set(get_neighbour_cells(valid_cell).values())
            #check if worker can reach the cell and the box nearby the cell can be pushed
            if can_go_there(warehouse, (valid_cell[1], valid_cell[0])) and temp_boxes_can_be_pushed != set():
                # worker can go to this cell which is nearby one box
                for temp_box in temp_boxes_can_be_pushed:
                    #check each possible pushable boxes nearby the worker
                    offset = temp_box[0]-valid_cell[0], temp_box[1]-valid_cell[1]
                    second_cell = temp_box[0]+offset[0], temp_box[1]+offset[1]
                    if second_cell not in dead_locks and second_cell not in warehouse.boxes and second_cell not in warehouse.walls:
                        worker_locations_nearby_boxes.add(valid_cell)
                        boxes_can_be_pushed.add(temp_box)
                
        return boxes_can_be_pushed, worker_locations_nearby_boxes
    
    def print_solution(self, goal_node):
        if goal_node == None:
            return ['Impossible']
        
        path = goal_node.path()
        solution = []
        for node in path:
            solution.append(node.action)
        
        solution.remove(None)
        print(solution)
    
    def return_solution(self, goal_node):
        if goal_node == None:
            return ['Impossible']
        
        path = goal_node.path()
        solution = []
        for node in path:
            solution.append(node.action)
        
        solution.remove(None)
        return solution
    
    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
