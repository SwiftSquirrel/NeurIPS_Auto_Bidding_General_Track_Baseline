from agent import Agent
import gymnasium as gym

env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
num_action = env.action_space.shape[0]
train = False
num_episodes = 50
model_path = './models'
device = 'cuda:0'

dqn_agent = Agent(lr=1e-3, discount_factor=0.999, num_action=num_action,
                  epsilon=1.0, batch_size=256, state_dim=state_dim, env_name=env_name, model_path=model_path, device=device)

if train:
    dqn_agent.train_model(env, num_episodes)
else:
    score = dqn_agent.test_model(render_mode='human', load_model=True)
    print(score)
