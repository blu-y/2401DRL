import pygame
import random

# Initialize the game
pygame.init()

# Set up the game window
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Tetris")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Define the game grid
grid_size = 30
grid_width = window_width // grid_size
grid_height = window_height // grid_size

# Define the Tetris shapes
shapes = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[1, 1, 1], [0, 1, 0]],
    [[1, 1, 1], [1, 0, 0]],
    [[1, 1, 1], [0, 0, 1]]
]

# Define the Tetris class
class Tetris:
    def __init__(self):
        self.grid = [[0] * grid_width for _ in range(grid_height)]
        self.current_shape = random.choice(shapes)
        self.current_x = grid_width // 2 - len(self.current_shape[0]) // 2
        self.current_y = 0

    def draw_grid(self):
        for y in range(grid_height):
            for x in range(grid_width):
                if self.grid[y][x] == 1:
                    pygame.draw.rect(window, WHITE, (x * grid_size, y * grid_size, grid_size, grid_size))
                else:
                    pygame.draw.rect(window, BLACK, (x * grid_size, y * grid_size, grid_size, grid_size), 1)

    def draw_shape(self):
        for y in range(len(self.current_shape)):
            for x in range(len(self.current_shape[0])):
                if self.current_shape[y][x] == 1:
                    pygame.draw.rect(window, BLUE, ((self.current_x + x) * grid_size, (self.current_y + y) * grid_size, grid_size, grid_size))

    def move_shape(self, dx, dy):
        if self.can_move(dx, dy):
            self.current_x += dx
            self.current_y += dy

    def can_move(self, dx, dy):
        for y in range(len(self.current_shape)):
            for x in range(len(self.current_shape[0])):
                if self.current_shape[y][x] == 1:
                    new_x = self.current_x + x + dx
                    new_y = self.current_y + y + dy
                    if new_x < 0 or new_x >= grid_width or new_y >= grid_height or self.grid[new_y][new_x] == 1:
                        return False
        return True

    def rotate_shape(self):
        rotated_shape = list(zip(*reversed(self.current_shape)))
        if self.can_rotate(rotated_shape):
            self.current_shape = rotated_shape

    def can_rotate(self, rotated_shape):
        for y in range(len(rotated_shape)):
            for x in range(len(rotated_shape[0])):
                if rotated_shape[y][x] == 1:
                    new_x = self.current_x + x
                    new_y = self.current_y + y
                    if new_x < 0 or new_x >= grid_width or new_y >= grid_height or self.grid[new_y][new_x] == 1:
                        return False
        return True

    def place_shape(self):
        for y in range(len(self.current_shape)):
            for x in range(len(self.current_shape[0])):
                if self.current_shape[y][x] == 1:
                    self.grid[self.current_y + y][self.current_x + x] = 1

    def check_lines(self):
        full_lines = []
        for y in range(grid_height):
            if all(self.grid[y]):
                full_lines.append(y)
        for line in full_lines:
            del self.grid[line]
            self.grid.insert(0, [0] * grid_width)

    def game_over(self):
        return any(self.grid[0])

    def update(self):
        self.draw_grid()
        self.draw_shape()
        pygame.display.update()

# Create the Tetris game instance
tetris = Tetris()

# Game loop
running = True
clock = pygame.time.Clock()
while running:
    clock.tick(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                tetris.move_shape(-1, 0)
            elif event.key == pygame.K_RIGHT:
                tetris.move_shape(1, 0)
            elif event.key == pygame.K_DOWN:
                tetris.move_shape(0, 1)
            elif event.key == pygame.K_UP:
                tetris.rotate_shape()

    if tetris.can_move(0, 1):
        tetris.move_shape(0, 1)
    else:
        tetris.place_shape()
        tetris.check_lines()
        if tetris.game_over():
            running = False
        else:
            tetris.current_shape = random.choice(shapes)
            tetris.current_x = grid_width // 2 - len(tetris.current_shape[0]) // 2
            tetris.current_y = 0

    window.fill(BLACK)
    tetris.update()

# Quit the game
pygame.quit()