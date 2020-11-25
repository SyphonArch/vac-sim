import pygame
from pygame.locals import *
import logic
import numpy as np
from random import random
import pickle
from plot import plot_history
import os

pygame.init()
pygame.display.set_caption('VAC_SIM by Syphon')
# Visual aspects
font1 = pygame.font.SysFont('Arial', 20)
font2 = pygame.font.SysFont('Arial', 30, bold=True)
colors = INFECTED, SUSCEPTIBLE, CURED, VACCINATED = (0, 255, 0), (255, 0, 0), (255, 255, 100), (0, 0, 255)
WHITE = (255, 255, 255)
BG = (209, 87, 87)
BLACK = (0, 0, 0)
radius = 5

fps = 600


def show_fps(screen, dimension, clock, font):
    observed_fps = int(clock.get_fps())
    fps_text = font.render(str(observed_fps), False, WHITE, BLACK)
    screen.blit(fps_text, (dimension - 40, 0))


def show_runinfo(screen, runinfo, font):
    fps_text = font.render(' ' + runinfo + ' ', False, WHITE, BG)
    screen.blit(fps_text, (40, 10))


def show_composition(screen, dimension, composition, font):
    composition_str = ' {:4d} |{:4d} |{:4d} |{:4d} '.format(*composition)
    fps_text = font.render(composition_str, False, WHITE, BG)
    screen.blit(fps_text, (dimension // 2 - 100, 10))


def run(dimension, composition, duration, acceleration, speed, box_size, infection_range, probability, efficacy,
        save_name, repeat=1, show_graph=False, autoexit=True):
    assert infection_range <= box_size
    sectors, rem = divmod(dimension, box_size)
    assert rem == 0
    if save_name:
        target_dir = f'results/{save_name}'
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode([dimension, dimension])
    running = True
    for t in range(repeat):
        locations = logic.rand_loc(sum(composition), dimension)
        person_state = []
        for i, cnt in enumerate(composition):
            person_state += [i] * cnt
        person_state = np.asarray(person_state)
        time_of_infection = np.zeros(len(locations))
        prev_offsets = np.zeros((len(locations), 2))
        history = [composition]
        frame = 0
        stop_sim = False
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
            if stop_sim:
                continue
            screen.fill(BLACK)
            frame += 1

            still_infected_filter = (person_state == 1) & (frame - time_of_infection <= duration)
            now_cured_filter = (person_state == 1) & (~still_infected_filter)
            person_state += now_cured_filter

            locations, prev_offsets = logic.rand_offset(locations, prev_offsets, dimension, acceleration, speed)

            for i, loc in enumerate(locations):
                color = colors[person_state[i]]
                pygame.draw.circle(screen, color, loc, radius)

            infected = locations[person_state == 1]
            infected_grid = logic.create_grid(infected, sectors, box_size)
            for i, person in enumerate(locations):
                if person_state[i] in (0, 3):
                    cnt = logic.radius_count(person, infected_grid, sectors, box_size, infection_range)
                    spread = logic.spread(cnt, probability)
                    if spread:
                        if person_state[i] == 0 or random() > efficacy:
                            person_state[i] = 1
                            time_of_infection[i] = frame

            current_composition = tuple(np.count_nonzero(person_state == i) for i in range(4))
            history.append(current_composition)
            if history[-1][1] == 0:
                if show_graph:
                    plot_history(history)
                if save_name:
                    with open(f'results/{save_name}/{t}.p', 'wb') as f:
                        pickle.dump(history, f)
                if not autoexit and t == repeat - 1:
                    stop_sim = True
                else:
                    break
            clock.tick(fps)
            show_fps(screen, dimension, clock, font1)
            runinfo = save_name + '_' + str(t)
            show_runinfo(screen, runinfo, font2)
            show_composition(screen, dimension, current_composition, font2)
            pygame.display.update()
        if not running:
            break


def semiconfigured_run(vaccination_rate, speed, probability, efficacy, save_name,
                       repeat=1, show_graph=False, autoexit=True):
    population = 1000
    initial_infected = 5
    non_infected = population - initial_infected
    vaccinated = int(non_infected * vaccination_rate)
    non_vaccinated = non_infected - vaccinated
    composition = (non_vaccinated, initial_infected, 0, vaccinated)
    run(dimension=1024,
        composition=composition,
        duration=100,
        acceleration=1,
        speed=speed,
        box_size=16,
        infection_range=16,
        probability=probability,
        efficacy=efficacy,
        save_name=save_name,
        repeat=repeat,
        show_graph=show_graph,
        autoexit=autoexit)


def two_param_run(vaccination_rate, efficacy, repeat=1, show_graph=False, autoexit=True):
    semiconfigured_run(vaccination_rate=vaccination_rate,
                       speed=2,
                       probability=0.077,
                       efficacy=efficacy,
                       save_name=f'{vaccination_rate}-{efficacy}',
                       repeat=repeat,
                       show_graph=show_graph,
                       autoexit=autoexit)


if __name__ == '__main__':
    vr = 1
    ec = 0.9
    two_param_run(vr, ec, repeat=100)
