import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move
from math import sqrt

total_frame_count = 0
CARDINAL_DIRECTIONS = [NORTH, EAST, SOUTH, WEST]
CAP = 255


class AutomatonsBot:
    """
    Class for instantiating a single bot for a game
    """

    def __init__(self, id):
        self.bot_id = id
        self.counter = 0
        self.move_count = 0
        self.strength_sum = 0
        self.has_made_contact = False

    def __find_nearest_enemy(self, game_map, start):
        """
        This method is used by default by the Overkill bot. It is
        a breadth-first search that radiates outward based on cardinal directions
        :param game_map: Current frame's game map
        :param start: Starting current player square
        :return: The nearest enemy direction
        """
        max_distance = min(game_map.width, game_map.height) / 2
        direction = NORTH
        for cardinal_direction in CARDINAL_DIRECTIONS:
            distance = 0
            current_square = start
            while current_square.owner == self.bot_id and distance < max_distance:
                distance += 1
                current_square = game_map.get_target(current_square, cardinal_direction)
            if distance < max_distance:
                direction = cardinal_direction
                max_distance = distance

        return direction

    def __heuristic(self, game_map, square):
        """
        Default heuristic used for Overkill bot, used to find the highest
        yield for unoccupied squares, or the sum of the effective Overkill for a given
         opponent square
        :param game_map: Current frame's game map
        :param square: Current square
        :return: Heuristic value, based on the occupant
        """
        if square.owner == 0 and square.strength > 0:
            return square.production / square.strength
        else:
            return sum(neighbor.strength
                       for neighbor in game_map.neighbors(square)
                       if neighbor.owner not in (0, self.bot_id))

    def __get_move(self, game_map, square):
        """
        Called by `get_moves` for each piece. Depending on the state of the game
        (early or late) and whether or not the piece has a target
        :param game_map: Current frame's game map
        :param square: Current player square
        :return: Move for a player square for a given frame
        """
        self.strength_sum += square.strength
        self.move_count += 1
        target, direction = max(((neighbor, direction)
                                 for direction, neighbor
                                 in enumerate(game_map.neighbors(square))
                                 if neighbor.owner != self.bot_id),
                                default=(None, None),
                                key=lambda t: self.__heuristic(game_map, t[0]))

        if target is not None and target.strength < square.strength:
            self.has_made_contact = target not in (0, self.bot_id)
            return Move(square, direction)
        elif square.strength < square.production * 5:
            return Move(square, STILL)

        border = any(neighbor.owner != self.bot_id for neighbor in game_map.neighbors(square))
        if not border:
            if self.__is_early_game():
                return Move(square, self.__find_max_production_direction(game_map, square))
            else:
                return Move(square, self.__find_nearest_enemy(game_map, square))
        else:
            # wait until we are strong enough to attack
            return Move(square, STILL)

    def __find_max_production_direction(self, game_map, start):
        """
        Modified breadth-first search to find areas of high production. This is used
        in early game before switching to an enemy-based breadth-first search
        :param game_map: Current game map for frame
        :param start: Starting square
        :return: Direction of max production
        """
        max_distance = min(game_map.width, game_map.height) / 2
        direction = NORTH
        max_production = start.production
        for cardinal_direction in CARDINAL_DIRECTIONS:
            distance = 0
            current_square = start
            while distance < max_distance:
                distance += 1
                current_square = game_map.get_target(current_square, cardinal_direction)
                if current_square.owner != self.bot_id and current_square.production > max_production:
                    max_production = current_square.production
                    direction = cardinal_direction

        return direction

    def __is_early_game(self):
        return self.counter / total_frame_count <= 0.33

    def get_moves(self, game_map):
        self.counter += 1
        self.move_count = 0
        return [self.__get_move(game_map, square) for square in game_map if square.owner == self.bot_id]


def main():
    my_id, game_map = hlt.get_init()
    hlt.send_init("MyBot")
    bot = AutomatonsBot(my_id)
    global total_frame_count
    # Dynamic total frame count according to Halite rules
    total_frame_count = 10 * sqrt(game_map.width * game_map.height)
    while True:
        game_map.get_frame()
        hlt.send_frame(bot.get_moves(game_map))


if __name__ == '__main__':
    main()
