import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
from queue import PriorityQueue

'''
The AutomatonsBot is based off of the Pythonic Overkill bot
'''

CARDINAL_DIRECTIONS = [NORTH, EAST, SOUTH, WEST]
CAP = 255


class AutomatonsBot:
    def __init__(self, id):
        self.bot_id = id
        self.count = 0
        self.last_percentage = 0

    def _find_nearest_enemy_direction(self, game_map, start):
        max_distance = min(game_map.width, game_map.height) / 2
        direction = NORTH
        square = start
        for cardinal_direction in CARDINAL_DIRECTIONS:
            distance = 0
            current_square = start
            while current_square.owner == self.bot_id and distance < max_distance:
                distance += 1
                current_square = game_map.get_target(current_square, cardinal_direction)
            if distance < max_distance:
                direction = cardinal_direction
                max_distance = distance
        return square, direction

    # def _find_highest_production(self, game_map, start):
    #     max_distance = min(game_map.width, game_map.height) / 2
    #     square = start
    #     for cardinal_direction in CARDINAL_DIRECTIONS:
    #         distance = 0
    #         current_square = start
    #         while current_square.owner == self.bot_id and distance < max_distance:
    #             distance += 1
    #             current_square = game_map.get_target(current_square, cardinal_direction)
    #         if distance < max_distance:
    #             if (current_square.production > start.production):
    #                 square = current_square
    #             max_distance = distance
    #     return square

    def _dijkstra(self, game_map, square):
        "See <http://www.redblobgames.com/pathfinding/a-star/introduction.html> for original impl."
        frontier = PriorityQueue()
        came_from = {}
        cost_so_far = {}

        goal = self._find_nearest_enemy(game_map, square)

        came_from[square] = None
        cost_so_far[square] = 0

        frontier.put((square, STILL), 0)

        direction = NORTH

        while not frontier.empty():
            current_square, square_direction = frontier.get()

            if current_square.owner != self.bot_id:
                direction = square_direction

            for cardinal_direction, next in enumerate(game_map.neighbors(current_square)):
                new_cost = cost_so_far[current_square] + self._cost(current_square, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + game_map.get_distance(current_square, goal)
                    frontier.put((next, cardinal_direction), priority)
                    came_from[next] = current_square

        return direction

    def _cost(self, current, next):

        is_less_than_max = current.production + next.production < CAP
        if current.production < next.production and is_less_than_max:
            return 1
        elif current.production > next.production and is_less_than_max:
            return 2
        elif not is_less_than_max:
            return 3
        else:
            return 0

    def _heuristic(self, game_map, square):
        if square.owner == 0 and square.strength > 0:
            return square.production / square.strength
        else:
            # return total potential damage caused by overkill when attacking this square
            return sum(neighbor.strength
                       for neighbor in game_map.neighbors(square)
                       if neighbor.owner not in (0, self.bot_id))

    def _get_move(self, game_map, square, counter):
        self.count += 1
        target, direction = max(((neighbor, direction)
                                 for direction, neighbor
                                 in enumerate(game_map.neighbors(square))
                                 if neighbor.owner != self.bot_id),
                                default=(None, None),
                                key=lambda t: self._heuristic(game_map, t[0]))

        if target is not None and target.strength < square.strength:
            return Move(square, direction)
        elif square.strength < square.production * 5:
            return Move(square, STILL)

        border = any(neighbor.owner != self.bot_id for neighbor in game_map.neighbors(square))
        if not border:
            if square.strength > 200 and self.last_percentage < 0.75:
                return Move(square, self._dijkstra(game_map, square))
            else:
                return Move(square, self._find_nearest_enemy_direction(game_map, square))
        else:
            # wait until we are strong enough to attack
            return Move(square, STILL)

    def get_moves(self, game_map, counter):
        self.count = 0
        return [self._get_move(game_map, square, counter) for square in game_map if square.owner == self.bot_id]


def main():
    my_id, game_map = hlt.get_init()
    hlt.send_init("MyBot")
    bot = AutomatonsBot(my_id)
    counter = 0
    while True:
        counter += 1
        game_map.get_frame()
        hlt.send_frame(bot.get_moves(game_map, counter))
        bot.last_percentage = bot.count / (game_map.width * game_map.height)


if __name__ == '__main__':
    main()
