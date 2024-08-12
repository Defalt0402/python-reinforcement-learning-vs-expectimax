from neural_net import *
from joblib import dump, load
import numpy as np
import os

class Expectimax_Agent():
    def __init__(self, game, depth=3):
        self.game = game
        self.depth = depth
        self.memo = {}

        self.memoFile = "memo/memo.pkl"
        if os.path.exists(self.memoFile):
            self.memo = load(self.memoFile)

    def save_memo(self):
        os.makedirs(os.path.dirname(self.memoFile), exist_ok=True)
        dump(self.memo, self.memoFile)

    def get_best_move(self):
        currentScore = self.game.score
        bestMove = None
        bestVal = float("-inf")
        for move in self.game.get_legal_moves():
            currentBoard = np.copy(self.game.board)
            self.game.expect_move(move)
            board_tuple = tuple(map(tuple, self.game.board))  # Convert to a hashable type
            if board_tuple in self.memo:
                val = self.memo[board_tuple]
            else:
                val = self.expectimax(self.game.board, self.depth, "chance", float("-inf"), float("inf"))
                self.memo[board_tuple] = val  

            if val > bestVal:
                bestVal = val
                bestMove = move
            self.game.board[:] = currentBoard
        self.game.score = currentScore
        return bestMove
    
    def expectimax(self, board, depth, player, alpha, beta):
        if depth == 0 or self.game.is_game_over():
            return self.game.evaluate_board()
        
        if player == "max":
            maxVal = float("-inf")
            for move in self.game.get_legal_moves():
                currentBoard = np.copy(board)
                self.game.expect_move(move)
                board_tuple = tuple(map(tuple, self.game.board))  # Convert to a hashable type
                if board_tuple in self.memo:
                    maxVal = max(maxVal, self.memo[board_tuple])
                else:
                    maxVal = max(maxVal, self.expectimax(self.game.board, depth-1, "chance", alpha, beta))
                self.game.board[:] = currentBoard
                alpha = max(alpha, maxVal)
                if beta <= alpha:  # Beta cut-off
                    break
            return maxVal
        elif player == "chance":
            emptyTiles = self.game.get_empty_tiles()
            numEmptyTiles = len(emptyTiles)
            if not emptyTiles or numEmptyTiles == 0:
                return self.expectimax(board, depth-1, "max", alpha, beta)

            expectedVal = 0
            for tile in emptyTiles:
                board[tile] = 2
                expectedVal += 0.9 * self.expectimax(board, depth-1, "max", alpha, beta)
                board[tile] = 4
                expectedVal += 0.1 * self.expectimax(board, depth-1, "max", alpha, beta)
                board[tile] = 0
            return expectedVal / numEmptyTiles