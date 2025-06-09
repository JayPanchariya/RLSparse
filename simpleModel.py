import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set seed
np.random.seed(42)
tf.random.set_seed(42)

# Define the reward function: L(x) = -(x + 3)^2 - 2
def L(x):
    return -(x + 3)**2 - 2

# Policy model: simple 2-layer MLP with outputs for mean and log_std
class SimplePolicy(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(32, activation='tanh')
        self.mean = tf.keras.layers.Dense(1)
        self.log_std = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = tf.nn.softplus(log_std) + 1e-5
        return mean, std

# Initialize model and optimizer
policy = SimplePolicy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

# Training config
T = 3
L_val = 10
epochs = 30
x0 = 0.0
trajectory_log = []

# Training loop
for epoch in range(epochs):
    trajectories = []
    rewards = []
    log_probs = []

    for l in range(L_val):
        x = tf.constant([[x0]], dtype=tf.float32)
        traj = [x0]
        traj_log_probs = []
        total_reward = 0.0

        for t in range(T):
            with tf.GradientTape() as tape:
                mean, std = policy(x)
                dist = tf.random.normal(shape=[1], mean=mean, stddev=std)
            a = dist.numpy()[0]
            log_prob = -((a - mean.numpy()[0])**2) / (2 * std.numpy()[0]**2) - np.log(std.numpy()[0]) - 0.5 * np.log(2 * np.pi)
            x = x + a
            traj.append(x.numpy()[0][0])
            total_reward += L(x.numpy()[0][0])
            traj_log_probs.append(log_prob)

        rewards.append(total_reward)
        log_probs.append(np.sum(traj_log_probs))
        if epoch >= epochs - 5:
            trajectory_log.append(traj)

    # Normalize rewards
    rewards = np.array(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # Compute and apply gradients
    with tf.GradientTape() as tape:
        loss = 0
        for i in range(L_val):
            loss += -log_probs[i] * rewards[i]
        loss /= L_val
    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))

# Plotting
x_vals = np.linspace(-6, 2, 400)
y_vals = L(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='L(x) = -(x + 3)^2 - 2', color='blue')

colors = plt.cm.plasma(np.linspace(0.3, 1.0, 5))
for i, traj in enumerate(trajectory_log[-5:]):
    plt.plot(traj, [L(x) for x in traj], 'o-', color=colors[i], label=f'Trajectory {epochs - 5 + i + 1}')

plt.title("REINFORCE-OPT with TensorFlow Low-Level API")
plt.xlabel("x")
plt.ylabel("L(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
tf_low_api_path = "/mnt/data/tensorflow_low_api_reinforce_opt.png"
plt.savefig(tf_low_api_path)
tf_low_api_path
