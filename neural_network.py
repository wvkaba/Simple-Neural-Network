import numpy as np

# The squisher: turns any number into a number between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The cards (inputs): each row is one card, each column is one box
X = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
])

# The right answers: one per card
y = np.array([[0, 1, 1, 0]]).T

# The dials (weights): random start between -1 and 1
np.random.seed(1)
weights = 2 * np.random.random((3, 1)) - 1

# Practice 10,000 times
for i in range(10000):
    total = np.dot(X, weights)        # add up the dials for boxes with stars
    guess = sigmoid(total)            # squish into 0 to 1
    error = y - guess                 # how wrong was the guess?
    weights = weights + np.dot(X.T, error)   # turn the dials

print("Dials after practice:")
print(weights.round(2))

print("Guesses after practice:")
print(guess.round(3))

# Test on a card the kid has never seen
new_card = np.array([[1, 0, 0]])
print("New card guess:")
print(sigmoid(np.dot(new_card, weights)).round(3))
