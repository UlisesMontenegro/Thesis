
import tensorflow as tf
import numpy as np
import gym

# Defining rates & factors
lr = 0.001 # Learning rate
gamma = 0.99 # Discount factor

# Creating the environment
env = gym.make('CartPole-v1')
inputShape = env.observation_space.shape[0] #4
numActions = env.action_space.n #2

# Defining the policy network
policyNetwork = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(inputShape,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(numActions, activation='softmax')
])

# Setting up optimizer and loss function
optimizer = tf.keras.optimizers.Adam(lr)
lossFunction = tf.keras.losses.SparseCategoricalCrossentropy()

# Setting up lists to store episode rewards and lengths
episodeRewards = []
episodeLengths = []

numEpisodes = 1000

# Training the agent using REINFORCE algorithm
for episode in range(numEpisodes):
    done = False
    print("Starting Episode: ", episode)
    # Reset the environment and get the initial state
    state = env.reset()[0]
    episodeReward = 0
    episodeLenght = 0

    # Keep track of the states, actions and rewards for each step
    states = []
    actions = []
    rewards = []

    # Run the episode
    while True:
        # Get the action probabilities from the policy network
        print("State: ", np.array([state]))
        actionProbs = policyNetwork.predict(np.array([state]))[0]
        print("Action Probs: ", actionProbs)

        # Choose an action based on the action probabilities
        action = np.random.choice(numActions, p=actionProbs)
        print("Action chosen: ", action)

        # Take the chosen action and observe the next state and reward
        nextState, reward, truncated, terminated, _ = env.step(action)
        if truncated or terminated:
            done = True
        
        # Store the current state, action and reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        # Update the current state and episode reward
        state = nextState
        episodeReward += reward
        episodeLenght += 1

        # End the episode if the environment is done
        if done:
            print('Episode {} done !!!!!'.format(episode))
            break

        # Calculate the discounted rewards for each step in the episode
        discountedRewards = np.zeros_like(rewards)
        runningTotal = 0
        for i in reversed(range(len(rewards))):
            runningTotal = runningTotal * gamma + rewards[i]
            discountedRewards[i] = runningTotal
        print("Discounted Rewards: ", discountedRewards)

        # Normalize the discounted rewards
        discountedRewards -= np.mean(discountedRewards)
        discountedRewards /= np.std(discountedRewards)

        # Convert the lists of states, actions and discounted rewards to tensors
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        discountedRewards = tf.convert_to_tensor(discountedRewards)

        # Train the policy network using REINFORCE
        with tf.GradientTape() as tape:
            # Get the action probabilities from the policy network
            actionProbs = policyNetwork(states)
            print("Action Probs: ", actionProbs)
            # Calculate loss
            loss = tf.cast(tf.math.log(tf.gather(actionProbs, actions, axis=1, batch_dims=1)),tf.float64)
            print("Raw loss: ", loss)
            loss = loss*discountedRewards
            loss = -tf.reduce_sum(loss)
            print("Final loss: ", loss)

        # Calculate the gradients and update the policy network
        grads = tape.gradient(loss, policyNetwork.trainable_variables)
        optimizer.apply_gradients(zip(grads, policyNetwork.trainable_variables))

        # Store the episode reward and length
        episodeRewards.append(episodeReward)
        print("Total Episode Reward: ", episodeReward)
        episodeLengths.append(episodeLenght)
        print("Total Episode Length: ", episodeLenght)

        policyNetwork.save('D:\Uli\FIUBA\Tesis\Practica\policyNetwork.keras')