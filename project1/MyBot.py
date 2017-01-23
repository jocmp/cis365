import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
from queue import PriorityQueue

'''
The AutomatonsBot is based off of the Pythonic Overkill bot
'''

CARDINAL_DIRECTIONS = [NORTH, EAST, SOUTH, WEST]


class AutomatonsBot:
    def __init__(self, id):
        self.bot_id = id

    def _find_nearest_enemy_direction(self, game_map, square):
        max_distance = min(game_map.width, game_map.height) / 2
        direction = NORTH
        for cardinal_direction in CARDINAL_DIRECTIONS:
            distance = 0
            current_square = square
            while current_square.owner == self.bot_id and distance < max_distance:
                distance += 1
                current_square = game_map.get_target(current_square, cardinal_direction)
            if distance < max_distance:
                direction = cardinal_direction
                max_distance = distance
        return direction

    def _dijkstra(self, game_map, square):
        frontier = PriorityQueue()
        frontier.put(square, 0)
        came_from = {}
        cost_so_far = {}
        came_from[square] = None
        cost_so_far[square] = 0

        direction = NORTH
        countdown = 100
        while not frontier.empty():
            current = frontier.get()
            countdown -= 1

            if current.owner not in (0, self.bot_id):
                break

            for cardinal_direction, next in enumerate(game_map.neighbors(current)):
                new_cost = cost_so_far[current] + self._cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost
                    frontier.put(next, priority)
                    came_from[next] = current
                    direction = cardinal_direction

        return direction

    def _cost(self, current, next):
        if current.production > next.production:
            return 1
        elif current.production < next.production:
            return 2
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

    def _get_move(self, game_map, square):
        target, direction = max(((neighbor, direction)
                                 for direction, neighbor
                                 in enumerate(game_map.neighbors(square))
                                 if neighbor.owner != self.bot_id),
                                default=(None, None),
                                key=lambda t: self._heuristic(game_map, t[0]))

        if target is not None and target.strength < square.strength:
            return Move(square, direction)
        elif square.strength < square.production * 10:
            return Move(square, STILL)

        border = any(neighbor.owner != self.bot_id for neighbor in game_map.neighbors(square))
        if not border:
            return Move(square, self._dijkstra(game_map, square))
        else:
            # wait until we are strong enough to attack
            return Move(square, STILL)

    def get_moves(self, game_map):
        return [self._get_move(game_map, square) for square in game_map if square.owner == self.bot_id]


def main():
    my_id, game_map = hlt.get_init()
    hlt.send_init("MyBot")
    bot = AutomatonsBot(my_id)
    while True:
        game_map.get_frame()
        hlt.send_frame(bot.get_moves(game_map))


if __name__ == '__main__':
    main()
