from flask import Flask, render_template, url_for, request, redirect, flash
import logging
import os
import copy
import math
import time
from typing import TypeVar
import itertools
import queue
from werkzeug.utils import secure_filename
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

UPLOAD_FOLDER = 'puzzle_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
secret_key = os.urandom(12).hex()
app.config['SECRET_KEY'] = secret_key

@app.route('/', methods=["GET", "POST"])   
def index():
    
    return render_template('index.html')

@app.route('/success', methods=["GET", "POST"])   
def solve():
    if request.method == 'POST':   
        file = request.files.get("file") 
        file_content = file.read().decode()
        # f = request.files['file'] 
        # f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        solver_method = request.form.get("solver_method")
        logging.info(solver_method)
        logging.info(file_content)
        valid, error = validate_input(file_content)
        logging.info("valid: ", valid)
        logging.info("error: ", error)
        
        if not valid:
          flash(error, 'error')
          return render_template('index.html')
        else:
          obstacles_list, storage_list, robot, box_list, size = initialize(file_content)

          solve_time = 0
          solution = []
          
          if solver_method == 'bfs':
              start = time.time()
              solution = bfs(robot, box_list, obstacles_list, storage_list, size);
              end = time.time()
              solve_time = end - start
          elif solver_method == "dfs":
              start = time.time()
              solution = dfs(robot, box_list, obstacles_list, storage_list, size);
              end = time.time()
              solve_time = end - start
          elif solver_method == "greedy":
              start = time.time()
              solution = greedy(robot, box_list, obstacles_list, storage_list, size);
              end = time.time()
              solve_time = end - start
          elif solver_method == "astar":
              start = time.time()
              solution = astar(robot, box_list, obstacles_list, storage_list, size);
              end = time.time()
              solve_time = end - start
          else:  # default is BFS
              start = time.time()
              solution = bfs(robot, box_list, obstacles_list, storage_list, size);
              end = time.time()
              solve_time = end - start

          
          return render_template('file_success.html',time=solve_time, solution=solution)

'''
  initialize: Check puzzle file contents to validate if there are any immediate errors found in the input
    parameters:
      file (String): String containing puzzle contents
    output:
      valid (boolean): returns True if no errors found, return False if there are errors found
      error (string): the error message decribing what is wrong with the puzzle input; returns empty string if the puzzle input is valid
'''      
def validate_input(file):
  valid = True
  error = ''
  
  # check if file content is empty
  if not file:
    valid = False
    error = 'Please upload a non-empty puzzle file'
    return valid, error
  
  # Check how many time the "R" (robot) appears in input puzzle
  robot_count = file.count('R') + file.count(".")
   
  if robot_count == 0:
    valid = False
    error = 'Puzzle Input needs to contain a Robot'
    return valid, error
  
  if robot_count > 1:
    valid = False
    error = 'Puzzle Input can contain ONLY 1 Robot'
    return valid, error
  
  # Check how many boxes and storage units appear in the input puzzle
  box_count = file.count("B")
  storage_count = file.count("S") + file.count(".")
  
  if box_count == 0:
    valid = False
    error = 'Puzzle Input must contain a Box'
    return valid, error
  
  if storage_count == 0:
    valid = False
    error = 'Puzzle Input must contain a Storage Unit'
    return valid, error
  
  if box_count != storage_count:
    valid = False
    error = 'Puzzle Input must contain the same number of Boxes as Storage Units'
    return valid, error
   
  
  return valid, error

'''
  initialize: Sets up Sokoban puzzle to solve based on contents of file
    parameters:
      file (String): String containing puzzle contents
    output:
      robot (tuple): Initial Location of Robot
      box_list (list of tuples): Initial Locations of Boxes
      storage_list (list of tuples): Locations of all Storage Spots
      obstacles_list (list of tuples): Locations of all Obstacles (Walls)
      size (tuple): Size of the Puzzle Grid
'''
def initialize(file):
    obstacles_list = []; # To Contain list of tuples with location of obstacles
    storage_list = [];  # To Contain list of tuples with location of storages
    box_list = [];  # To Contain list of tuples with location of boxes
    robot = (0, 0)  # To Contain a tuple with location of robot
    size = (0, 0)   # Tuple, total size of playing grid
    
    # with open(file) as f:  # Take in the input text file
    #     lines = f.readlines()
    lines = file.split("\n")

    # Set size of playing grid
    m = len(lines)
    logging.info("lines: ", lines)
    n = len(lines[0].replace("\\n", ""))
    size = (m , n)

    # Print out Lines to see what grid looks like
    for line in lines:
        line = line.replace("\\n", "")
        print(line)


    # Initialize list of Obstacles, Robot, Boxes, and Storage
    for x in range(len(lines)):
        line = lines[x]
        line = line.replace("\\n", "")
        
        for y in range(len(line)):

            if (line[y] == "R"):  # 'R' indicates Robot
                robot = (x,y);
            
            if (line[y] == "S"):  # 'R' indicates Storage
                storage = (x,y)
                storage_list.append(storage)
            
            if (line[y] == "O"):  # 'O' indicates Obstacle (Wall)
                obstacle = (x,y)
                obstacles_list.append(obstacle)

            if (line[y] == "B"):  # 'B' indicates Box
                box = (x,y)
                box_list.append(box)

            if (line[y] == "."):  # '.' indicates Robot and Storage in same location
                robot = (x,y);
                storage = (x,y)
                storage_list.append(storage)

    obstacles_list, storage_list, robot, box_list, size
    # logging.info(obstacles_list)
    # logging.info(storage_list)
    # logging.info(robot)
    # logging.info(box_list)
    # logging.info(size)
    return obstacles_list, storage_list, robot, box_list, size

'''
  ---- *start* Priority Queue (FIFO) Implementation *start* ---

  The following code was borrowed from the source here: 
  https://www.redshiftzero.com/priority-queue/?fbclid=IwAR0HGy1l4xvyi0oflTaXD8kv9PPhgmctQ2eoHRwgQ44GKnghUWF-g-m79kg
  This code was used to create Priority Queues (for Greedy and A* Search), where items of the same Priority are pulled in the order they were created in.
'''
QueueJobType = TypeVar('QueueJobType', bound='QueueJob')
class QueueJob():
    def __init__(self, order_number: int, task: str) -> None:
        self.order_number = order_number
        self.task = task

    def __lt__(self, other: QueueJobType) -> bool:
        '''
        We need to use the order_number key to break ties to ensure
        that objects are retrieved in FIFO order.
        '''
        return self.order_number < other.order_number

    def __repr__(self) -> str:
        return self.task


class PossibleMoves():
    def __init__(self):
        self.order_number = itertools.count()
        self.queue = queue.PriorityQueue()

    def add_task(self, priority: int, task: str):
        current_order_number = next(self.order_number)
        task = QueueJob(current_order_number, task)
        self.queue.put((priority, task))

'''
  ---- *end* Priority Queue (FIFO) Implementation *end* ---
'''

'''
  successor_func: Get the next set of Valid Moves from the Robot and Boxes current location
    parameters:
      robot_temp (tuple): Current Location of Robot
      box_list_temp (list of tuples): Current Locations of Boxes
      obstacles_list (list of tuples): Locations of all Obstacles (Walls)
      storage_list (list of tuples): Locations of all Storage spots
      size (tuple): Size of the Puzzle Grid
    output:
      valid_moves (list of strings): List of the next valid directions the Robot can move (i.e. ["D", "R", "L", "U"])
'''
def successor_func(robot_temp, box_list_temp, obstacles_list, storage_list, size):
  valid_moves = [];

  # Coordinates of next possible moves (Up, Down , Left, Right)
  move_up = (robot_temp[0] - 1, robot_temp[1])
  move_down = (robot_temp[0] + 1, robot_temp[1])
  move_left = (robot_temp[0], robot_temp[1] - 1)
  move_right = (robot_temp[0], robot_temp[1] + 1)

  # Check which directions the Robot can move
  if (check_move(move_up, "U", box_list_temp, obstacles_list, storage_list, size)):
    valid_moves.append("U")
  if (check_move(move_down, "D", box_list_temp, obstacles_list, storage_list, size)):
    valid_moves.append("D")
  if (check_move(move_left, "L", box_list_temp, obstacles_list, storage_list, size)):
    valid_moves.append("L")
  if (check_move(move_right, "R", box_list_temp, obstacles_list, storage_list, size)):
    valid_moves.append("R")

  return valid_moves

'''
  check_move: Verify if it is possible for the Robot to move into a given location (used by the successor_func function)
    parameters:
      move_coord (tuple): Location of Next Possible Move of Robot
      direction (string): Direction that the Robot is moving in (Only Possible Values: "D", "R", "L", "U")
      box_list_temp (list of tuples): Current Locations of Boxes
      obstacles_list (list of tuples): Locations of all Obstacles (Walls)
      storage_list (list of tuples): Locations of all Storage spots
      size (tuple): Size of the Puzzle Grid
    output:
      valid_move (Boolean): Value is True if the Robot can move in this Direction, Value is False if the Robot cannot
'''
def check_move(move_coord, direction, box_list_temp, obstacles_list, storage_list, size):
  valid_move = False;

  # Coordinate for moving must be within the grid
  if (move_coord[0] > 0 and move_coord[0] < size[0] and move_coord[1] > 0 and move_coord[1] < size [1]):
    # Coordinate for moving cannot be the same coordinate as an obstacle
    if (move_coord not in obstacles_list):
      # If coordinate for moving contains a Box
      if (move_coord in box_list_temp):
        # Need to Check if Box can be moved in the same direction as the Robot
        for box_temp in box_list_temp:
          if (move_coord == box_temp): # only care about the box that's in the same space the Robot is attempting to move to
            valid_box_move = False
            if direction == "U":
              box_move = (box_temp[0] - 1, box_temp[1])
              valid_box_move = check_box_move(box_move, box_temp, box_list_temp, obstacles_list, storage_list, size)
            elif direction == "D":
              box_move = (box_temp[0] + 1, box_temp[1])
              valid_box_move = check_box_move(box_move, box_temp, box_list_temp, obstacles_list, storage_list, size)
            elif direction == "L":
              box_move = (box_temp[0], box_temp[1] - 1)
              valid_box_move = check_box_move(box_move, box_temp, box_list_temp, obstacles_list, storage_list, size)
            elif direction == "R":
              box_move = (box_temp[0], box_temp[1] + 1)
              valid_box_move = check_box_move(box_move, box_temp, box_list_temp, obstacles_list, storage_list, size)
            
            if (valid_box_move):
              valid_move = True
      else:
        valid_move = True

  return valid_move;

'''
  check_box_move: Verify if it is possible for the Box to move into a given location (used by the check_move function)
    parameters:
      move_coord (tuple): Location of Next Possible Move of the Box
      current_box (tuple): Location of where the Box currently is
      box_list_temp (list of tuples): Current Locations of Boxes
      obstacles_list (list of tuples): Locations of all Obstacles (Walls)
      storage_list (list of tuples): Locations of all Storage spots
      size (tuple): Size of the Puzzle Grid
    output:
      box_valid_move (Boolean): Value is True if the Box can move into this location, Value is False if the Box cannot
'''
def check_box_move(move_coord, current_box, box_list_temp, obstacles_list, storage_list, size):
  box_list_temp2 = copy.deepcopy(box_list_temp);
  box_list_temp2.remove(current_box);
  box_valid_move = False;

  # print("move_coord[0] > 0 and move_coord[0] < size[0] and move_coord[1] > 0 and move_coord[1] < size [1]: " , move_coord[0] > 0 and move_coord[0] < size[0] and move_coord[1] > 0 and move_coord[1] < size [1]);

  # Coordinate for moving must be within the grid
  if (move_coord[0] > 0 and move_coord[0] < size[0] and move_coord[1] > 0 and move_coord[1] < size [1]):
    # Coordinate for moving cannot be the same coordinate as an obstacle or another box
    if (move_coord not in obstacles_list and move_coord not in box_list_temp2):
        # box_valid_move = True;

        # Check if the box will be cornered in the new location 

        if move_coord in storage_list:  # It doesn't matter if the box is cornered if it's going into storage
          box_valid_move = True;
        else:

          # Get the coordinates around the new location of the Box
          block_up = (move_coord[0] - 1, move_coord[1]);
          block_down = (move_coord[0] + 1, move_coord[1]);
          block_left = (move_coord[0], move_coord[1] - 1);
          block_right = (move_coord[0], move_coord[1] + 1);


          up_blocked = False;
          down_blocked = False;
          left_blocked = False;
          right_blocked = False;

          num_blocked = 0;
          
          # Check if there are any obstacles or boxes in the locations around the box
          if (block_up in obstacles_list or block_up in box_list_temp2):
            up_blocked = True;
          if (block_down in obstacles_list or block_down in box_list_temp2):
            down_blocked = True;
          if (block_left in obstacles_list or block_left in box_list_temp2):
            left_blocked = True;
          if (block_right in obstacles_list or block_right in box_list_temp2):
            right_blocked = True;


          up_left_blocked = False;
          up_right_blocked = False;
          down_left_blocked = False;
          down_right_blocked = False;

          if (up_blocked and left_blocked):
            in_between = (block_left[0] - 1 , block_left[1]);
            # If the coordinate between the diagonal is within the grid
            if (in_between[0] < size[0] and in_between[1] < size[1]):
              # If the coordinate between the diagonal is blocked by a wall or a box
              if (in_between not in obstacles_list):
                if (block_left not in box_list_temp2 and block_up not in box_list_temp2):
                  up_left_blocked = True;# then the diagonal is blocked
            else:
              up_left_blocked = True; # then the diagonal is blocked


          if (up_blocked and right_blocked):
            in_between = (block_right[0] - 1 , block_right[1]);
            # If the coordinate between the diagonal is within the grid
            if (in_between[0] < size[0] and in_between[1] < size[1]):
              # If the coordinate between the diagonal is blocked by a wall or a box
              if (in_between not in obstacles_list):
                if (block_right not in box_list_temp2 and block_up not in box_list_temp2):
                  up_right_blocked = True;# then the diagonal is blocked
            else:
              up_right_blocked = True; # then the diagonal is blocked


          if (down_blocked and left_blocked):
            in_between = (block_left[0] + 1, block_left[1]);
            # If the coordinate between the diagonal is within the grid
            if (in_between[0] < size[0] and in_between[1] < size[1]):
              # If the coordinate between the diagonal is blocked by a wall or a box
              if (in_between not in obstacles_list):
                if (block_left not in box_list_temp2 and block_down not in box_list_temp2):
                  down_left_blocked = True;# then the diagonal is blocked
            else:
              down_left_blocked = True; # then the diagonal is blocked


          if (down_blocked and right_blocked):
            in_between = (block_right[0] + 1, block_right[1]);
            # If the coordinate between the diagonal is within the grid
            if (in_between[0] < size[0] and in_between[1] < size[1]):
              # If the coordinate between the diagonal is blocked by a wall or a box
              if (in_between not in obstacles_list):
                if (block_right not in box_list_temp2 and block_down not in box_list_temp2):
                  down_right_blocked = True;# then the diagonal is blocked
            else:
              down_right_blocked = True; # then the diagonal is blocked


          # If the box isn't blocked in by any diagonals, then we can successfully move the box to this position
          if ( not up_left_blocked and not up_right_blocked and not down_left_blocked and not down_right_blocked):
            box_valid_move = True;


  return box_valid_move;

'''
  goal_state: Check if the game is won
    parameters:
      box_list_temp (list of tuples): Current Locations of Boxes
      storage_list (list of tuples): Locations of all Storage spots
    output:
      game_won (Boolean): Value is True if all Boxes are in a separate Storage spot
'''
def goal_state (box_list_temp, storage_list):
  game_won = False

  occupied_storage_set = set();

  for box in box_list_temp:
    for storage in storage_list:
      if (box == storage):
        occupied_storage_set.add(storage)

  storage_list_set = set(storage_list)

  if occupied_storage_set == storage_list_set:
    game_won = True

  return game_won;

'''
  update_map: Update the game (robot and box locations) based on a list of moves
    parameters:
      moves (list of strings): List of direction the Robot can move (a possible solution to winning the game)
      robot (tuple): Initial Location of Robot
      box_list (list of tuples): Initial Locations of Boxes
    output:
      robot_new (tuple): New Location of Robot
      box_list_new (list of tuples): New Locations of Boxes
'''
def update_map(moves, robot, box_list):
  robot_new = (robot[0] , robot[1])
  box_list_new = copy.deepcopy(box_list)

  for move in moves:
    ## Set the Robot's new coordinate
    if move == "U":
      robot_new = (robot_new[0] - 1, robot_new[1])
    elif move == "D":
      robot_new = (robot_new[0] + 1, robot_new[1])
    elif move == "L":
      robot_new = (robot_new[0], robot_new[1] - 1)
    elif move == "R":
      robot_new = (robot_new[0], robot_new[1] + 1)

    # if the robot has pushed a box, we need to update the box's coordinates
    for i in range(len(box_list_new)):
      box = box_list_new[i]
      if robot_new == box_list_new[i]:
        if move == "U":
          box_list_new[i] = (box[0] - 1, box[1])
        elif move == "D":
          box_list_new[i] = (box[0] + 1, box[1])
        elif move == "L":
          box_list_new[i] = (box[0], box[1] - 1)
        elif move == "R":
          box_list_new[i] = (box[0], box[1] + 1)
      

  
  return robot_new, box_list_new

'''
  man_dist_heur: Simple Manhattan Distance Heuristic, 
                 does not take into consideration any obstacles between box and storage, 
                 multiple boxes can be stored in same storage location
    parameters:
      box_list (list of tuples): Current Locations of Boxes
      storage_list (list of tuples): Locations of all Storage Spots
      size (tuple): Size of the Puzzle Grid
    output:
      man_dist_sum (Integer): Sum of Manhattan Distance for Each Unstored Box and the Storage Closest to it
'''
def man_dist_heur(box_list, storage_list, size):
  man_dist_sum = 0;

  for box in box_list:
    closest = float('inf')
    for storage in storage_list:
      md = abs(box[0] - storage[0]) + abs(box[1] - storage[1]);
      if md < closest:
        closest = md;
    
    man_dist_sum = man_dist_sum + closest;
  
  return man_dist_sum;

'''
  man_dist_heur_impr: Improved Manhattan Distance Heuristic, 
                 takes into consideration some obstacles between box and storage, 
                 only one box can be stored in same storage location
    parameters:
      box_list (list of tuples): Current Locations of Boxes
      storage_list (list of tuples): Locations of all Storage Spots
      obstacles_list (list of tuples): Locations of all Obstacles (Walls)
      size (tuple): Size of the Puzzle Grid
    output:
      man_dist_sum (Decimal): Sum of Manhattan Distance for Each Unstored Box and the Storage Closest to it
'''
def man_dist_heur_impr (box_list, storage_list, obstacles_list, size):
  man_dist_sum = 0;

  storage_dict = {};  # keep track of which storage is closest to each box

  for box in box_list:
    closest = float('inf')
    for storage in storage_list:

      # if this is already the closest storage for a previous box
      if (storage in storage_dict.values()):
        continue; # then we need to consider a different storage for this box

      # Get Manhattan Distance to the storage
      md = abs(box[0] - storage[0]) + abs(box[1] - storage[1]);

      # Check for some potential obstacles in the path between the box and storage
      middle = (math.ceil( ( box[0] + storage[0] ) / 2) , math.ceil( ( box[1] + storage[1] ) / 2 ));
      corner1 = (box[0], storage[1]);
      corner2 = (box[1], storage[0]);


      obstacle_bias = 0;

      if (middle in obstacles_list or middle in box_list):
        obstacle_bias = obstacle_bias + 0.5;

      if (corner1 in obstacles_list or corner1 in box_list):
        obstacle_bias = obstacle_bias + 0.5;
      
      if (corner2 in obstacles_list or corner2 in box_list):
        obstacle_bias = obstacle_bias + 0.5;


      # If there are some potential obstacles in the way, add to the manhattan distance
      md = md + obstacle_bias;


      if md < closest:
        closest = md;
        storage_dict[box] = storage;  # set this as the closest storage for this box


    man_dist_sum = man_dist_sum + closest;
  

  return man_dist_sum;


'''
  bfs: Breadth First Search to find solution for Sokoban puzzle
    parameters:
      robot (tuple): Initial Location of Robot
      box_list (list of tuples): Initial Locations of Boxes
      storage_list (list of tuples): Locations of all Storage Spots
      obstacles_list (list of tuples): Locations of all Obstacles (Walls)
      size (tuple): Size of the Puzzle Grid
    output:
      man_dist_sum (Decimal): Sum of Manhattan Distance for Each Unstored Box and the Storage Closest to it
'''
def bfs (robot, box_list, obstacles_list, storage_list, size):
  game_won = False
  solution = []

  history = {}
  k = 0

  history[robot] = [box_list]

  possible_moves = [];
  initial_valid_moves = successor_func(robot, box_list, obstacles_list, storage_list, size)
  for move in initial_valid_moves:
    move_list = []
    move_list.append(move)
    possible_moves.append(move_list)
  print("initial, possible_moves: " , possible_moves)

  while not game_won and len(possible_moves) > 0:
    # Get set of possible moves to win game
    moves = possible_moves.pop(0) # Use Queue (FIFO)

    # Get updated map of game using new set of possible moves
    robot_new, box_list_new = update_map(moves, robot, box_list)


    # Check if we have already tried this solution before
    visited = False
    if (robot_new in history.keys()):
      box_hist = history.get(robot_new)
      if box_list_new in box_hist:
        visited = True


    # If we haven't tried this before
    if (not visited):

      # Add this to history
      if robot_new in history:
        box_hist = history.get(robot_new)
        box_hist.append(box_list_new)
      else:
        history[robot_new] = [box_list_new]

      # Check if game is won with the set of possible moves
      game_won = goal_state (box_list_new, storage_list)

      # if the new set of possible moves has not won the game
      if (not game_won):
        # get list of next valid moves based on current set of possible moves
        valid_moves = successor_func(robot_new, box_list_new, obstacles_list, storage_list, size)


        ## add new sets of possible moves to the queue
        ##    ie if current possible moves list for the solution is ["U", "D"]
        ##        and next valid moves are "L" or "R"
        ##        then the following are added to the BFS queue: ["U", "D", "L"], ["U", "D", "R"]

        if len(valid_moves) > 0:
          for vmove in valid_moves:
            new_moves = copy.deepcopy(moves)
            new_moves.append(vmove)
            possible_moves.append(new_moves)
      else:
        solution.extend(moves)
    

    k = k + 1
  

  print("bfs, k: " , k)

  return solution

'''
  dfs: Depth First Search to find solution for Sokoban puzzle
    parameters:
      robot (tuple): Initial Location of Robot
      box_list (list of tuples): Initial Locations of Boxes
      storage_list (list of tuples): Locations of all Storage Spots
      obstacles_list (list of tuples): Locations of all Obstacles (Walls)
      size (tuple): Size of the Puzzle Grid
    output:
      man_dist_sum (Decimal): Sum of Manhattan Distance for Each Unstored Box and the Storage Closest to it
'''
def dfs (robot, box_list, obstacles_list, storage_list, size):
  game_won = False
  solution = []

  history = {}
  k = 0

  history[robot] = [box_list]

  possible_moves = [];
  initial_valid_moves = successor_func(robot, box_list, obstacles_list, storage_list, size)
  for move in initial_valid_moves:
    move_list = []
    move_list.append(move)
    possible_moves.append(move_list)
  print("initial, possible_moves: " , possible_moves)

  while not game_won and len(possible_moves) > 0:
    # Get set of possible moves to win game
    moves = possible_moves.pop()  # use stack (LIFO) instead of queue

    # Get updated map of game using new set of possible moves
    robot_new, box_list_new = update_map(moves, robot, box_list)

    # Check if we have already tried this solution before
    visited = False
    if (robot_new in history.keys()):
      box_hist = history.get(robot_new)
      if box_list_new in box_hist:
        visited = True

    # If we haven't tried this before
    if (not visited):

      # Add this to history
      if robot_new in history:
        box_hist = history.get(robot_new)
        box_hist.append(box_list_new)
      else:
        history[robot_new] = [box_list_new]

      # Check if game is won with the set of possible moves
      game_won = goal_state (box_list_new, storage_list)

      # if the new set of possible moves has not won the game
      if (not game_won):
        # get list of next valid moves based on current set of possible moves
        valid_moves = successor_func(robot_new, box_list_new, obstacles_list, storage_list, size)

        if len(valid_moves) > 0:
          for vmove in valid_moves:
            new_moves = copy.deepcopy(moves)
            new_moves.append(vmove)
            possible_moves.append(new_moves)
      else:
        solution.extend(moves)
    
    k = k + 1 # Keep Track of Number of iterations to complete Search
  
  print("dfs, k: " , k)
  return solution

'''
  greedy: Greedy Search to find solution for Sokoban puzzle
    parameters:
      robot (tuple): Initial Location of Robot
      box_list (list of tuples): Initial Locations of Boxes
      storage_list (list of tuples): Locations of all Storage Spots
      obstacles_list (list of tuples): Locations of all Obstacles (Walls)
      size (tuple): Size of the Puzzle Grid
    output:
      man_dist_sum (Decimal): Sum of Manhattan Distance for Each Unstored Box and the Storage Closest to it
'''
def greedy (robot, box_list, obstacles_list, storage_list, size):
  game_won = False
  solution = []

  history = {}
  k = 0

  history[robot] = [box_list]

  # Add initial set of moves (empty list) and related Manhattan Distance to Priority Queue
  possible_moves = PossibleMoves()
  md = man_dist_heur_impr (box_list, storage_list, obstacles_list, size)
  possible_moves.add_task(md, '')


  while not game_won and not possible_moves.queue.empty():
    # Get set of possible moves to win game
    md, moves_qj = possible_moves.queue.get();  # Priority Queue (FIFO, based on Priority)
    moves_string = moves_qj.task;

    moves = []
    if len(moves_string) > 0:
      moves = moves_string.split(",")

    # Get updated map of game using new set of possible moves
    robot_new, box_list_new = update_map(moves, robot, box_list)

    # Check if game is won with the set of possible moves
    game_won = goal_state (box_list_new, storage_list)

    # if the new set of possible moves has not won the game
    if (not game_won):
      # get list of next valid moves based on current set of possible moves
      valid_moves = successor_func(robot_new, box_list_new, obstacles_list, storage_list, size)

      ## add new sets of possible moves to the queue
      ##    ie if current possible moves list for the solution is ["U", "D"]
      ##        and next valid moves are "L" or "R"
      ##        then the following are added to the BFS queue: ["U", "D", "L"], ["U", "D", "R"]

      if len(valid_moves) > 0:
        for vmove in valid_moves:
          new_moves = copy.deepcopy(moves)
          new_moves.append(vmove)

          # Get updated map of game using this set of new valid moves
          robot_vnew, box_list_vnew = update_map(new_moves, robot, box_list)

          # Check if we have already tried this solution before
          visited = False
          if (robot_vnew in history.keys()):
            box_hist = history.get(robot_vnew)
            if box_list_vnew in box_hist:
              visited = True


          if not visited:
            # Add this to history
            if robot_vnew in history:
              box_hist = history.get(robot_vnew)
              box_hist.append(box_list_vnew)
            else:
              history[robot_vnew] = [box_list_vnew]

            # Get Manhattan Distance based Heuristic for this new set of Valid Moves
            md = man_dist_heur_impr (box_list_vnew, storage_list, obstacles_list, size)

            # Add this new set of moves and man dist to Possible Moves Priority Queue
            new_moves_string = ','.join(new_moves);
            possible_moves.add_task(md, new_moves_string)
    else:
      solution.extend(moves)
    

    k = k + 1 # Keep Track of Number of iterations to complete Search
  
  print("greedy, k: " , k)
  return solution

'''
  astar: A* Search to find solution for Sokoban puzzle
    parameters:
      robot (tuple): Initial Location of Robot
      box_list (list of tuples): Initial Locations of Boxes
      storage_list (list of tuples): Locations of all Storage Spots
      obstacles_list (list of tuples): Locations of all Obstacles (Walls)
      size (tuple): Size of the Puzzle Grid
    output:
      man_dist_sum (Decimal): Sum of Manhattan Distance for Each Unstored Box and the Storage Closest to it
'''
def astar (robot, box_list, obstacles_list, storage_list, size):
  game_won = False
  solution = []

  history = {}
  k = 0

  history[robot] = [box_list]

  # Add initial set of moves (empty list) and related Manhattan Distance to Priority Queue
  possible_moves = PossibleMoves()
  md = man_dist_heur_impr (box_list, storage_list, obstacles_list, size)
  possible_moves.add_task(md, '')


  while not game_won and not possible_moves.queue.empty():
    # Get set of possible moves to win game
    md, moves_qj = possible_moves.queue.get();  # Priority Queue (FIFO, based on Priority)
    moves_string = moves_qj.task;

    moves = []
    if len(moves_string) > 0:
      moves = moves_string.split(",")


    # Get updated map of game using new set of possible moves
    robot_new, box_list_new = update_map(moves, robot, box_list)


    # Check if game is won with the set of possible moves
    game_won = goal_state (box_list_new, storage_list)

    # if the new set of possible moves has not won the game
    if (not game_won):
      # get list of next valid moves based on current set of possible moves
      valid_moves = successor_func(robot_new, box_list_new, obstacles_list, storage_list, size)

      ## add new sets of possible moves to the queue
      ##    ie if current possible moves list for the solution is ["U", "D"]
      ##        and next valid moves are "L" or "R"
      ##        then the following are added to the BFS queue: ["U", "D", "L"], ["U", "D", "R"]

      if len(valid_moves) > 0:
        for vmove in valid_moves:
          new_moves = copy.deepcopy(moves)
          new_moves.append(vmove)

          # Get updated map of game using this set of new valid moves
          robot_vnew, box_list_vnew = update_map(new_moves, robot, box_list)


          # Check if we have already tried this solution before
          visited = False
          if (robot_vnew in history.keys()):
            box_hist = history.get(robot_vnew)
            if box_list_vnew in box_hist:
              visited = True


          if not visited:
            # Add this to history
            if robot_vnew in history:
              box_hist = history.get(robot_vnew)
              box_hist.append(box_list_vnew)
            else:
              history[robot_vnew] = [box_list_vnew]

            # Get Manhattan Distance based Heuristic for this new set of Valid Moves
            md_g = man_dist_heur_impr (box_list_vnew, box_list, obstacles_list, size) # get the path cost from beginning to now
            md_h = man_dist_heur_impr (box_list_vnew, storage_list, obstacles_list, size) # get the path cost from now to goal

            md = md_g + md_h

            # print("astar, md: ", md)

            # Add this new set of moves and man dist to Possible Moves Priority Queue
            new_moves_string = ','.join(new_moves);
            possible_moves.add_task(md, new_moves_string)

    else:
      solution.extend(moves)
    

    k = k + 1 # Keep Track of Number of iterations to complete Search
  
  print("astar, k: " , k)
  return solution

if __name__ == '__main__':   
    app.run(debug=True)
