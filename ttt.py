import random
import math

class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_legal_moves()
        self.move = move
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def add_child(self, state, move):
        child_node = Node(state, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child_node)
        return child_node
    
    def best_child(self, c_param=1.4):
        choices_weights = []
        for child in self.children:
            if child.visits == 0:
                choices_weights.append(float('inf'))  # Ensure unvisited nodes are explored first
            else:
                exploit = child.wins / child.visits
                explore = c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
                choices_weights.append(exploit + explore)
        return self.children[choices_weights.index(max(choices_weights))]
    
    def update(self, result):
        self.visits += 1
        if result == 0:
            self.wins += 0.5
        else:
            self.wins += result

class TicTacToeState:
    def __init__(self, board=None, player=1):
        self.board = board or [[' ' for _ in range(3)] for _ in range(3)]
        self.player = player
        self.last_move = None
    
    def get_legal_moves(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r][c] == ' ']
    
    def do_move(self, move):
        r, c = move
        self.board[r][c] = 'X' if self.player == 1 else 'O'
        self.player *= -1
        self.last_move = move
  
    def is_terminal(self):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != ' ':
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != ' ':
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return True
        # Check for a draw (full board)
        if all(self.board[r][c] != ' ' for r in range(3) for c in range(3)):
            return True
        
        return False
    
    def result(self, player):
        # Determine the result of the game from the perspective of the player
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] == 'X':
                return 1 if player == 1 else -1
            if self.board[0][i] == self.board[1][i] == self.board[2][i] == 'X':
                return 1 if player == 1 else -1

        if self.board[0][0] == self.board[1][1] == self.board[2][2] == 'X':
            return 1 if player == 1 else -1

        if self.board[0][2] == self.board[1][1] == self.board[2][0] == 'X' :
            return 1 if player == 1 else -1


        # Determine the result of the game from the perspective of the player
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] == 'O':
                return 1 if player == -1 else -1
            if self.board[0][i] == self.board[1][i] == self.board[2][i] == 'O':
                return 1 if player == -1 else -1

        if self.board[0][0] == self.board[1][1] == self.board[2][2] == 'O':
            return 1 if player == -1 else -1

        if self.board[0][2] == self.board[1][1] == self.board[2][0] == 'O':
            return 1 if player == -1 else -1

        if all(self.board[r][c] != ' ' for r in range(3) for c in range(3)):
            return 0  # Draw
        
        return None  # Ongoing game

        
    def clone(self):
        return TicTacToeState([row[:] for row in self.board], self.player)
    
    def display(self):
        for row in self.board:
            print('|', ' | '.join(row), '|')
        print()

class MCTS:
    def __init__(self, iterations):
        self.iterations = iterations

    def search(self, initial_state):
        root = Node(initial_state)
 
        for x in range(self.iterations):
            node = root
            state = initial_state.clone()
            
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                state.do_move(node.move)

            # Expansion
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                state.do_move(move)
                node = node.add_child(state, move)
                
            # Simulation
            if not state.is_terminal():
                while not state.is_terminal():
                    state.do_move(random.choice(state.get_legal_moves()))


            # Backpropagation
            result = state.result(-1)
            while node:
                if result is not None:
                    node.update(result)
                node = node.parent

        s = sorted(root.children, key=lambda c: c.visits)
        for i in s:
            print(f"Move: {i.move}, Wins: {i.wins}, Visits: {i.visits}, Win rate: {i.wins/i.visits}")
            
        best_move = sorted(root.children, key=lambda c: c.visits)[-1].move
        print(f"Best move after {self.iterations} iterations: {best_move}")
        return best_move

def play_game():
    initial_state = TicTacToeState()
    mcts = MCTS(iterations=50000)
    initial_state.player = 1
    while not initial_state.is_terminal():
        initial_state.display()

        if initial_state.player == 1:
            # Human's turn
            move = None
            while move not in initial_state.get_legal_moves():
                try:
                    row = int(input("Enter row (0, 1, 2): "))
                    col = int(input("Enter column (0, 1, 2): "))
                    move = (row, col)
                except ValueError:
                    print("Invalid input. Please enter numbers 0, 1, or 2.")
            initial_state.do_move(move)
        else:
            # AI's turn
            move = mcts.search(initial_state)
            print(f"AI chose move: {move}")
            initial_state.do_move(move)

    initial_state.display()
    result = initial_state.result(1)
    
    if result == 1:
        print("Human (X) wins!")
    elif result == -1:
        print("AI (O) wins!")
    elif result == 0:
        print("It's a draw!")
    else:
        print('Error: Game result is None.')

# Start the game
play_game()


