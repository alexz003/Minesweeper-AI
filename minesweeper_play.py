import minesweeper_game
from minesweeper_models import deepq_network, alexnet3
import tflearn as tf
import numpy as np
import time


VERSION = 'only_unpicked'
BATCH_SIZE = 10000

T_STEPS = 5
LR = 1e-3
GAMMA = 0.99
EPSILON = 0.8

N = 20
MODEL_NAME = 'ms_ai_{}k_{}_{}_{}_{}_{}.model'.format(int(T_STEPS*BATCH_SIZE/1000), VERSION, N, LR, GAMMA, EPSILON)         


def calc_reward(game, input_board, x, y, num_moves):

        if game.checkBomb((x,y)):
                return 0
        if (game.board == input_board).all():
                return 0
        
        score = 0
        for y in range(N):
                for x in range(N):
                        if input_board[x,y] == -1:
                                continue
                        score += 1 + input_board[x,y]
         
        return (score - num_moves)*game.checkBomb((x, y))

def main():
        game = minesweeper_game.Minesweeper()
        input_data = game.start_game()
        done = False
        
        network = deepq_network(N, N, LR, outputs=N**2)

        game_count = 0
        
        end_time = None
        start_time = time.time()
        for t in range(T_STEPS):

                batch = []
                if end_time == None:
                        print("Gathering batch data...{}/{}".format(t+1,T_STEPS))
                else:
                        print("Gathering batch data...{}/{}".format(t+1,T_STEPS))
                        print("ETA: {:5d}s".format(int(end_time-start_time)*(T_STEPS-t)))
                        start_time = time.time()
                        
                for b in range(BATCH_SIZE):
                        eps = t*BATCH_SIZE + b + 1
                        # Choose position
                        x, y = 0, 0
                        if np.random.rand() < EPSILON:
                                while(True):
                                        x = np.random.randint(0, N)                                
                                        y = np.random.randint(0, N)
                                        if input_data[x,y] < 0:
                                                break
                                
                        else:
                                actions = np.argsort(network.predict(input_data.reshape(1, N, N, 1)))
                                for i in actions[0]:
                                        x = int(i/N)
                                        y = i % N
                                        if input_data[x,y] < 0:
                                                break


                        new_input = game.send_move(x, y)
                        done = (game.gameStatus() != 0)
                        
                        game_count += 1
                        reward = calc_reward(game, input_data, x, y, game_count)
                        batch.append((input_data, (x, y), reward, new_input, done))

                        # Update input data
                        input_data = new_input

                        
                        # Create a new game
                        if game.gameStatus() != 0:
                                game = minesweeper_game.Minesweeper()
                                input_data = game.start_game()
                                game_count = 0
                                done = False

                print("Training batch data...")

                np.random.shuffle(batch)

                train_batch = batch[:-100]
                test_batch = batch[-100:]

                train_shape = (len(train_batch), N, N, 1)
                test_shape = (len(test_batch), N, N, 1)
                #(len(test_batch),) + input_data.shape[1:]
                
                train_inputs = np.zeros(train_shape)
                train_targets = np.zeros((len(train_batch), N*N))
                
                test_inputs = np.zeros(test_shape)
                test_targets = np.zeros((len(test_batch), N*N))

                train_batch = batch[:-100]
                test_batch = batch[-100:]

                # Prepare training data
                for i in range(len(train_batch)):
                        input_data = train_batch[i][0]
                        action = train_batch[i][1]
                        reward = train_batch[i][2]
                        new_input = train_batch[i][3]
                        done = train_batch[i][4]

                        train_inputs[i] = input_data.reshape(1, N, N, 1)
                        train_targets[i] = network.predict(input_data.reshape(1, N, N, 1))
                        Q_p = network.predict(new_input.reshape(1, N, N, 1))

                        if done:
                               train_targets[i, action] = reward
                        else:
                               train_targets[i, action] = reward + GAMMA*np.max(Q_p)

                # Prepare testing data
                for i in range(len(test_batch)):
                        input_data = test_batch[i][0]
                        action = test_batch[i][1]
                        reward = test_batch[i][2]
                        new_input = test_batch[i][3]
                        done = test_batch[i][4]

                        test_inputs[i] = input_data.reshape(1, N, N, 1)
                        test_targets[i] = network.predict(input_data.reshape(1, N, N, 1))
                        Q_p = network.predict(new_input.reshape(1, N, N, 1))

                        if done:
                               test_targets[i, action] = reward
                        else:
                               test_targets[i, action] = reward + GAMMA*np.max(Q_p)

                              

                # Fit model to inputs/outputs
                network.fit({'input' : train_inputs}, {'targets' : train_targets}, n_epoch=1,
                            validation_set=({'input' : test_inputs}, {'targets' : test_targets}),
                            show_metric=True, run_id=MODEL_NAME)

                network.save(MODEL_NAME)
                               
                end_time = time.time()
                               

        # Play the game
        done = False
        
        game = minesweeper_game.Minesweeper()
        input_data = game.start_game()
        count = 0
        final_reward = 0
        while game.gameStatus() == 0:
                x, y = 0, 0
                actions = np.argsort(network.predict(input_data.reshape(1, N, N, 1)))
                for i in actions[0]:
                        x = int(i/N)
                        y = i % N
                        if input_data[x,y] < 0:
                                break
                
                new_input = game.send_move(x, y)
                done = (game.gameStatus() != 0)
                                        
                count += 1
                reward = calc_reward(game, input_data, x, y, count)

                final_reward += reward

                input_data = new_input
                game.printBoard()
                time.sleep(1)

        print("Final reward: {}!".format(final_reward))



main()
