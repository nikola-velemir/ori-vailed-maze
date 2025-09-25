from game import Game

if __name__ == '__main__':
    game = Game(board_file='../boards/board.json', default_search='UCS', cell_size=100)
    game.run()
