# -*- coding: utf-8 -*-


import pygame

from pygame.locals import *

import sys

import tensorflow as tf
print('test1')
import cv2

import random
print('test1')
import numpy as np
print('test1')
from collections import deque

print(tf.__version__)
print('test1')
BLACK = (0, 0, 0)

WHITE = (255, 255, 255)

SCREEN_SIZE = [320, 400]

BAR_SIZE = [20, 5]

BALL_SIZE = [15, 15]

MOVE_STAY = [1, 0, 0]

MOVE_LEFT = [0, 1, 0]

MOVE_RIGHT = [0, 0, 1]

LEARN_RATE = 0.99

INIT_ESPTION = 1.0

FINAL_ESPTION = 0.05

EXPLORE = 50000

OBSERVE = 5000

REPLAY_MEMORY = 500000

BATCH = 100

print('test2')
class Game(object):

    def __init__(self):

        pygame.init()

        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode(SCREEN_SIZE)

        pygame.display.set_caption('Simple Game')

        self.ball_pos_x = SCREEN_SIZE[0] // 2 - BALL_SIZE[0] / 2

        self.ball_pos_y = SCREEN_SIZE[1] // 2 - BALL_SIZE[1] / 2

        # ball移动方向

        self.ball_dir_x = -1  # -1 = left 1 = right

        self.ball_dir_y = -1  # -1 = up   1 = down

        self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])

        self.score = 0

        self.bar_pos_x = SCREEN_SIZE[0] // 2 - BAR_SIZE[0] // 2

        self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1] - BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])

    def bar_move_left(self):

        self.bar_pos_x = self.bar_pos_x - 2

    def bar_move_right(self):

        self.bar_pos_x = self.bar_pos_x + 2

    def run(self, action):

        # pygame.mouse.set_visible(0) # make cursor invisible

        # bar_move_left = False

        # bar_move_right = False

        while True:

            ''' for event in pygame.event.get():

                if event.type == QUIT:

                    pygame.quit()

                    sys.exit()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 鼠标左键按下(左移)

                    bar_move_left = True

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: # 鼠标左键释放

                    bar_move_left = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3: #右键

                    bar_move_right = True

                elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:

                    bar_move_right = False

'''

            if action == MOVE_LEFT:
                self.bar_move_left()

            if action == MOVE_RIGHT:

                self.bar_move_right()

            else:

                pass

            if self.bar_pos_x < 0:
                self.bar_pos_x = 0

            if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
                self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]

            self.screen.fill(BLACK)

            self.bar_pos.left = self.bar_pos_x

            pygame.draw.rect(self.screen, WHITE, self.bar_pos)

            self.ball_pos.left += self.ball_dir_x * 2

            self.ball_pos.bottom += self.ball_dir_y * 3

            pygame.draw.rect(self.screen, WHITE, self.ball_pos)

            if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1] + 1):
                self.ball_dir_y = self.ball_dir_y * -1

            if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
                self.ball_dir_x = self.ball_dir_x * -1

            reward = 0

            if self.bar_pos.top <= self.ball_pos.bottom and (
                    self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):

                self.score += 1

                reward = 1

                print("Score: ", self.score, end='\r')

            elif self.bar_pos.top <= self.ball_pos.bottom and (
                    self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):

                self.score = 0

                print("Game Over: ", self.score)

                reward = -1

            pygame.display.update()

            self.clock.tick(60)

            MyGame_image = pygame.surfarray.array3d(pygame.display.get_surface())

            return reward, MyGame_image


output = 3

input_image = tf.placeholder("float", [None, 80, 100, 4])

action = tf.placeholder("float", [None, 3])


def convolutional_neural_network(input_image):
    weights = {'w_conv1': tf.Variable(tf.zeros([8, 8, 4, 32])),

               'w_conv2': tf.Variable(tf.zeros([4, 4, 32, 64])),

               'w_conv3': tf.Variable(tf.zeros([3, 3, 64, 64])),

               'w_fc4': tf.Variable(tf.zeros([3456, 784])),

               'w_out': tf.Variable(tf.zeros([784, output]))}

    biases = {'b_conv1': tf.Variable(tf.zeros([32])),

              'b_conv2': tf.Variable(tf.zeros([64])),

              'b_conv3': tf.Variable(tf.zeros([64])),

              'b_fc4': tf.Variable(tf.zeros([784])),

              'b_out': tf.Variable(tf.zeros([output]))}

    conv1 = tf.nn.relu(
        tf.nn.conv2d(input_image, weights['w_conv1'], strides=[1, 4, 4, 1], padding="VALID") + biases['b_conv1'])

    conv2 = tf.nn.relu(
        tf.nn.conv2d(conv1, weights['w_conv2'], strides=[1, 2, 2, 1], padding="VALID") + biases['b_conv2'])

    conv3 = tf.nn.relu(
        tf.nn.conv2d(conv2, weights['w_conv3'], strides=[1, 1, 1, 1], padding="VALID") + biases['b_conv3'])

    conv3_flat = tf.reshape(conv3, [-1, 3456])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, weights['w_fc4']) + biases['b_fc4'])

    output_layer = tf.matmul(fc4, weights['w_out']) + biases['b_out']

    return output_layer


def train_neural_network(imput_image):
    predict_action = convolutional_neural_network(input_image)

    argmax = tf.placeholder("float", [None, output])

    gt = tf.placeholder("float", [None])

    action = tf.reduce_sum(tf.multiply(predict_action, argmax), reduction_indices=1)

    cost = tf.reduce_mean(tf.square(action - gt))

    optimizer = tf.train.AdadeltaOptimizer(1e-6).minimize(cost)

    game = Game()

    D = deque()

    _, image = game.run(MOVE_STAY)

    image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)

    ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    input_image_data = np.stack((image, image, image, image), axis=2)

    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()

        n = 0

        epsilon = INIT_ESPTION

        while True:

            action_t = predict_action.eval(feed_dict={input_image: [input_image_data]})[0]

            argmax_t = np.zeros([output], dtype=np.int)

            if (random.random() <= INIT_ESPTION):

                maxIndex = random.randrange(output)



            else:

                maxIndex = np.armax(action_t)

            argmax_t[maxIndex] = 1

            if epsilon > FINAL_ESPTION:
                epsilon -= (INIT_ESPTION - FINAL_ESPTION) / EXPLORE

            reward, image = game.run(list(argmax_t))

            image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)

            ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

            image = np.reshape(image, (80, 100, 1))

            input_image_datal = np.append(image, input_image_data[:, :, 0:3], axis=2)

            D.append((input_image_data, argmax_t, reward, input_image_datal))

            if len(D) > REPLAY_MEMORY:
                D.popleft()

            if n > OBSERVE:

                minibatch = random.sample(D, BATCH)

                input_image_data_batch = [d[0] for d in minibatch]

                argmax_batch = [d[1] for d in minibatch]

                reward_batch = [d[2] for d in minibatch]

                input_image_data1_batch = [d[3] for d in minibatch]

                gt_batch = []

                out_batch = predict_action.eval(feed_dict={input_image: input_image_data1_batch})

                for i in range(0, len(minibatch)):
                    gt_batch.append(reward_batch[i] + LEARN_RATE * np.max(out_batch[i]))

                optimizer.run(feed_dict={gt: gt_batch, argmax: argmax_batch, input_image: input_image_data_batch})

            input_image_data = input_image_datal

            n = n + 1

            if n % 10000 == 0:
                # saver.save(sess, 'C:\\Users\\hasee\\game.cpk', global_step=n)

                print(n, "epsilon:", epsilon, " ", "action:", maxIndex, " ", "reward: ", reward)


train_neural_network(input_image)
