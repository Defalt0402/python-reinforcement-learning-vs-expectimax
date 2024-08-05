from neural_net import *
import numpy as np

class Expectimax_Agent():
    def __init__(self, game, depth=3):
        self.game = game
        self.depth = depth

    def get_best_move(self):
        currentScore = self.game.score
        bestMove = None
        bestVal = float("-inf")
        for move in self.game.get_legal_moves():
            currentBoard = self.game.board.copy()
            self.game.expect_move(move)
            val = self.expectimax(self.game.board, self.depth, "chance")
            if val > bestVal:
                bestVal = val
                bestMove = move
            self.game.board = currentBoard
        self.game.score = currentScore
        return bestMove
    
    def expectimax(self, board, depth, player):
        if depth == 0 or self.game.is_game_over():
            return self.game.evaluate_board()
        
        if player == "max":
            maxVal = float("-inf")
            for move in self.game.get_legal_moves():
                currentBoard = board.copy()
                self.game.expect_move(move)
                maxVal = max(maxVal, self.expectimax(self.game.board, depth-1, "chance"))
                self.game.board = currentBoard
            return maxVal
        elif player == "chance":
            emptyTiles = self.game.get_empty_tiles()
            if not emptyTiles:
                return self.expectimax(board, depth-1, "max")

            expectedVal = 0
            for tile in emptyTiles:
                board[tile] = 2
                expectedVal += 0.9 * self.expectimax(board, depth-1, "max")
                board[tile] = 4
                expectedVal += 0.1 * self.expectimax(board, depth-1, "max")
                board[tile] = 0
            return expectedVal / len(emptyTiles)