from src.visuals.game.game import Game

if __name__ == '__main__':
    game = Game(board_file='boards/open_board.json', cell_size=100)
    game.run()
