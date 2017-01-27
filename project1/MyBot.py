import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move
from queue import PriorityQueue
from math import sqrt

'''
The AutomatonsBot is based off of the Pythonic Overkill bot
'''

total_frame_count = 0
CARDINAL_DIRECTIONS = [NORTH, EAST, SOUTH, WEST]
CAP = 255


class AutomatonsBot:
    def __init__(self, id):
        self.bot_id = id
        self.counter = 0
        self.move_count = 0
        self.strength_sum = 0
        self.has_made_contact = False
        self.last_leader_direction = STILL

    def _find_nearest_enemy(self, game_map, start):
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

    def _dijkstra(self, game_map, square):
        "See <http://www.redblobgames.com/pathfinding/a-star/introduction.html> for original impl."
        frontier = PriorityQueue()
        came_from = {}
        cost_so_far = {}

        came_from[square] = None
        cost_so_far[square] = 0

        goal = self._find_nearest_enemy(game_map, square)[0]

        frontier.put((square, STILL), 0)

        direction = NORTH

        while not frontier.empty():
            current_square, square_direction = frontier.get()

            if current_square.owner != self.bot_id:
                direction = square_direction

            for cardinal_direction, next in enumerate(game_map.neighbors(current_square)):
                new_cost = cost_so_far[current_square] + self._heuristic(game_map,current_square)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + game_map.get_distance(current_square, goal)
                    frontier.put((next, cardinal_direction), priority)
                    came_from[next] = current_square

        return direction

    def _heuristic(self, game_map, square):
        if square.owner == 0 and square.strength > 0:
            return square.production / square.strength
        else:
            # return total potential damage caused by overkill when attacking this square
            return sum(neighbor.strength
                       for neighbor in game_map.neighbors(square)
                       if neighbor.owner not in (0, self.bot_id))

    def _get_move(self, game_map, square):
        self.strength_sum += square.strength
        self.move_count += 1
        target, direction = max(((neighbor, direction)
                                 for direction, neighbor
                                 in enumerate(game_map.neighbors(square))
                                 if neighbor.owner != self.bot_id),
                                default=(None, None),
                                key=lambda t: self._heuristic(game_map, t[0]))

        if target is not None and target.strength < square.strength:
            self.has_made_contact = target not in (0, self.bot_id)
            return Move(square, direction)
        elif square.strength < square.production * 5:
            return Move(square, STILL)

        border = any(neighbor.owner != self.bot_id for neighbor in game_map.neighbors(square))
        if not border:
            if not self.has_made_contact:
                return Move(square, self._dijkstra(game_map, square))
            else:
                return Move(square, self._find_nearest_enemy(game_map, square)[1])
        else:
            # wait until we are strong enough to attack
            return Move(square, STILL)

    def is_early_game(self):
        return self.counter / total_frame_count <= 0.33

    def get_moves(self, game_map):
        self.counter += 1
        self.move_count = 0
        return [self._get_move(game_map, square) for square in game_map if square.owner == self.bot_id]


def main():
    my_id, game_map = hlt.get_init()
    hlt.send_init("MyBot")
    bot = AutomatonsBot(my_id)
    global total_frame_count
    total_frame_count = 10 * sqrt(game_map.width * game_map.height)
    while True:
        game_map.get_frame()
        hlt.send_frame(bot.get_moves(game_map))


if __name__ == '__main__':
    main()
