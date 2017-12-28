#!/usr/bin/python
# -*- coding:utf8 -*-

"""
    @author:xiaotian zhao
    @time:12/28/17
"""

from __future__ import print_function

import heapq
import copy


def manhattan_distance(current, goal):
    current = [int(x) for x in current.split()]
    goal = [int(x) for x in goal.split()]

    distances = []

    for i in xrange(3):
        for j in xrange(3):
            for k in xrange(3):
                for l in xrange(3):
                    if current[i*3 + j] != -1 and current[i*3 + j] == goal[k*3 + l]:
                        distances.append(abs(k-i)+abs(l-j))

    return sum(distances)


def is_same(current, goal):
    return current == goal


def reconstract_path(came_from, current_node):

    if current_node in came_from:
        p = reconstract_path(came_from, came_from[current_node])
        return p + '\n' + current_node
    else:
        return current_node


def get_candicate_items(state):
    state = [int(x) for x in state.split()]
    candicate_items = []

    blank_x, blank_y = -1, -1
    for i in xrange(len(state)):
        if state[i] == -1:
            blank_x = i / 3
            blank_y = i % 3

    # print(blank_x, blank_y)
    if blank_x + 1 < 3:
        new_state = copy.deepcopy(state)
        tmp = new_state[(blank_x + 1) * 3 + blank_y]
        new_state[(blank_x + 1) * 3 + blank_y] = -1
        new_state[blank_x * 3 + blank_y] = tmp
        candicate_items.append(' '.join([str(x) for x in new_state]))

    if blank_x - 1 >= 0:
        new_state = copy.deepcopy(state)
        tmp = new_state[(blank_x - 1) * 3 + blank_y]
        new_state[(blank_x - 1) * 3 + blank_y] = -1
        new_state[blank_x * 3 + blank_y] = tmp
        candicate_items.append(' '.join([str(x) for x in new_state]))

    if blank_y + 1 < 3:
        new_state = copy.deepcopy(state)
        tmp = new_state[blank_x * 3 + blank_y + 1]
        new_state[blank_x * 3 + blank_y + 1] = -1
        new_state[blank_x * 3 + blank_y] = tmp
        candicate_items.append(' '.join([str(x) for x in new_state]))

    if blank_y - 1 >= 0:
        new_state = copy.deepcopy(state)
        tmp = new_state[blank_x * 3 + blank_y - 1]
        new_state[blank_x * 3 + blank_y - 1] = -1
        new_state[blank_x * 3 + blank_y] = tmp
        candicate_items.append(' '.join([str(x) for x in new_state]))

    return candicate_items


def astar(start, goal):
    closedset = []
    openset = []
    g_score = dict()
    came_from = dict()
    h_score = dict()

    g_score[start] = 0
    h_score[start] = manhattan_distance(start, goal)
    f_score = g_score[start] + h_score[start]

    heapq.heappush(openset, (f_score, start))

    while len(openset) > 0:
        _, current = heapq.heappop(openset)
        if is_same(current, goal):
            return reconstract_path(came_from, goal)

        closedset.append(current)

        for candicate_item in get_candicate_items(current):
            if candicate_item in closedset:
                continue

            tentative_g_score = g_score[current] + 1
            if candicate_item not in openset:
                tentative_is_better = True
            elif tentative_g_score < g_score[candicate_item]:
                tentative_is_better = True
            else:
                tentative_is_better = False

            if tentative_is_better:
                came_from[candicate_item] = current
                g_score[candicate_item] = tentative_g_score
                h_score[candicate_item] = manhattan_distance(candicate_item, goal)
                f_score = g_score[candicate_item] + h_score[candicate_item]
                heapq.heappush(openset, (f_score, candicate_item))

    return False


if __name__ == '__main__':
    start_state = '7 2 4 5 -1 6 8 3 1'
    goal_state = '-1 1 2 3 4 5 6 7 8'

    paths = astar(start_state, goal_state).split('\n')
    for path in paths:
        print_node = path.split()
        for i in xrange(3):
            tmp = []
            for j in xrange(3):
                tmp.append(print_node[i*3 + j])
            print(' '.join(tmp))
        print('')
