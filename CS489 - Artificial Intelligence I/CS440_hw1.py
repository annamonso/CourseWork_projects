import sys

class Game:
    def __init__(self, player1, player2, mode, algo):
        self.player1 = player1
        self.player2 = player2
        self.mode = mode
        self.algo = algo
    
    def toMove(self,state):
        board, turn = state
        return turn
        
    def update_move(self, state, move):
        board, turn = state
        new_board = board.copy()
        new_board[move-1] = turn
       
        next_player = self.player1 if turn == self.player2 else self.player2
        return (new_board, next_player)

    def show_board(self,state):
        board, turn = state
        print(f" {board[0]} | {board[1]} | {board[2]} ")
        print("---+---+---")
        print(f" {board[3]} | {board[4]} | {board[5]} ")
        print("---+---+---")
        print(f" {board[6]} | {board[7]} | {board[8]} ")

    def isTerminal(self, state):
        board, turn = state
        g = board
        
        # filas
        for i in [0, 3, 6]:
            if g[i] != " " and g[i] == g[i+1] == g[i+2]:
                return g[i]
        # columnas
        for i in [0, 1, 2]:
            if g[i] != " " and g[i] == g[i+3] == g[i+6]:
                return g[i]
        # diagonales
        if g[0] != " " and g[0] == g[4] == g[8]:
            return g[0]
        if g[2] != " " and g[2] == g[4] == g[6]:
            return g[2]
        # empate
        if " " not in g:
            return "Tie"
        return None
        
        return None

    def utility(self, result, player_max):
        if result == "Tie":
            return 0
        elif result == player_max:
            return 1
        else:
            return -1
    
    def actions(self,state):
        board, turn = state
        actions_available = []
        for i in range(9):
            if board[i]== " ":
                actions_available.append(i+1)
        return actions_available

    
    
    def result(self, state, a):
        board, turn = state
        new_board = board.copy()
        new_board[a-1] = turn
        if turn == self.player2:
            next_player = self.player1
        else:
            next_player = self.player2
        return (new_board, next_player)
    
    def check_game(self,state):
        status = game.isTerminal(state)
        if status == "Tie":
            print("TIE")
            return 1
        elif status == "X":
            print("X WON, O LOST")
            return 1
        elif status == "O":
            print("O WON, X LOST")
            return 1
        else: 
            return 0

#######################################
########## MINIMAX - SEARCH ###########
#######################################

def min_value(g, state):
    result = g.isTerminal(state)
    if result is not None:
        return g.utility(result, g.player2), None,1
    
    v = float("inf")
    best_move = None
    tree_nodes = 1
    for a in g.actions(state):
        v2, a2, subtrees = max_value(g, g.result(state, a))
        tree_nodes += subtrees
        if v2 < v:
            v, best_move = v2, a
    return v, best_move, tree_nodes
        

def max_value(g, state):
    result = g.isTerminal(state)
    if result is not None:
        return g.utility(result, g.player2), None,1
    
    v = float("-inf")
    best_move = None
    tree_nodes = 1
    for a in g.actions(state):
        v2, a2, subtrees = min_value(g, g.result(state, a))
        tree_nodes += subtrees
        if v2 > v:
            v, best_move = v2, a
    return v, best_move, tree_nodes

def miniMaxsearch(game,state):
    player = game.toMove(state)
    value, move,num_trees = max_value(game, state)
    return move,num_trees 

#######################################
######### ALPHA-BETA-SEARCH ###########
#######################################

def min_value_AlphaBeta(g, state, alpha, beta):
    result = g.isTerminal(state)
    if result is not None:
        return g.utility(result, g.player2), None,1
    
    v = float("inf")
    best_move = None
    tree_nodes = 1
    for a in g.actions(state):
        v2, a2, subtrees = max_value_AlphaBeta(g, g.result(state, a), alpha, beta)
        tree_nodes += subtrees
        if v2 < v:
            v, best_move = v2, a
            beta = min(beta, v)
        if v <= alpha:
            return v, best_move,tree_nodes
    return v, best_move, tree_nodes

def max_value_AlphaBeta(g, state, alpha, beta):
    result = g.isTerminal(state)
    if result is not None:
        return g.utility(result, g.player2), None,1
    
    v = float("-inf")
    best_move = None
    tree_nodes = 1
    
    for a in g.actions(state):
        v2, a2, subtrees = min_value_AlphaBeta(g, g.result(state, a), alpha, beta)
        tree_nodes += subtrees
        if v2 > v:
            v, best_move = v2, a
            alpha = max(alpha,v)
        if v >= beta:
            return v,best_move,tree_nodes
            
    return v, best_move, tree_nodes

   
def miniMaxAlphaBeta(game, state):
    player = game.toMove(state)
    v, move, num_trees = max_value_AlphaBeta(game,state, float("-inf"), float("inf"))
    return move,num_trees 




#######################################
################ MAIN #################
#######################################

if len(sys.argv) == 4:
    algorithm_type = int(sys.argv[1])
    player1 = sys.argv[2]
    mode = int(sys.argv[3])
else:
    print("ERROR: Not enough/too many/illegal input arguments.")
    
print("Monso Rodriguez, Anna, A20653296 solution:")
if algorithm_type==1:
    print("Algorithm: MiniMax-Search")
elif algorithm_type==2:
    print("Algorithm: MiniMax with alpha-beta prunning")
else:
    print("Algorithm not allowed")

print(f"Fist: {player1}")
if player1=="X":
    player2 = "O"
else:
    player2 ="X"
    
if mode==1:
    print("Mode: human versus computer")
elif mode==2:
    print("Mode: computer versus computer")
    
game = Game(player1, player2, mode, algorithm_type)

state = ([" " for _ in range(9)], player1)

if mode==1:
    while True:
        board, turn = state
        if turn == player1:
            print(f"X’s move. What is your move (possible moves at the moment are: {str(game.actions(state))} | enter 0 to exit the game)?")
            move = int(input())
            if move==0:
                break
            elif move not in game.actions(state):
                continue
            
            state = game.update_move(state, move)
            game.show_board(state)
        else:
            if algorithm_type==1:
                move, num_trees = miniMaxsearch(game, state)
            else: 
                move, num_trees = miniMaxAlphaBeta(game,state)
            if move is None:
                break   

            state = game.update_move(state, move)
            game.show_board(state)
            print(f"{player2}'s select move: {move}. Number of search tree node generated: {num_trees}")

        check = game.check_game(state)
        if check == 1:
            break
                
if mode == 2:
    while True:
        board, turn = state
        print(f"Computer playing as {turn} is playing...")
        if algorithm_type==1:
            move, num_trees = miniMaxsearch(game, state)
                
        else: 
            move, num_trees = miniMaxAlphaBeta(game,state)
        
        if move is None:
            break   

        state = game.update_move(state, move)
        game.show_board(state)
        print(f"{turn}'s select move: {move}. Number of search tree node generated: {num_trees}")
        
        check = game.check_game(state)
        if check == 1:
            break
        
                
            
    