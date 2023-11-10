''' 调试日志
1. 适配版本
https://medium.com/mlearning-ai/how-to-install-tensorflow-2-x-with-cuda-and-cudnn-on-ubuntu-20-04-lts-b73c209d8e88

2. 要用save_npz_dict 保存模型而不是 save_npz; 加载时同理
3. 用 APF 代替部分随机探索效果要好很多
4. 加入了PER: (https://blog.csdn.net/abcdefg90876/article/details/106270925)， 也可以只用Original Replay Buffer
5. 超参数参考模块: hyper parameters
'''

import argparse
import os
import random
import time
import gym
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorlayer as tl


from gym_examples.wrappers import RelativePosition
from prioritized_memory import Memory


'''
    GridWorld-v0:
    
    @Action -- 0 right, 1 up, 2 left, 3 down
    @Observation -- {[x1, y1], [x2, y2], 25 vector(6,)}, agent_loc, target_loc and surrounding states.
    @Info -- distance between agent and target
'''



parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='train or test', default='train')
parser.add_argument(
    '--save_path', default='dqn_variants', help='folder to save if mode == train else model path,'
    'qnet will be saved once target net update'
)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--noisy_scale', type=float, default=1e-2)
parser.add_argument('--disable_double', action='store_false', default=True)
parser.add_argument('--disable_dueling', action='store_false', default=False)
args = parser.parse_args()





if args.mode == 'train':
    os.makedirs(args.save_path, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)  # reproducible
noise_scale = args.noisy_scale
double = not args.disable_double
dueling = not args.disable_dueling

env = gym.make('gym_examples/GridWorld-v0', render_mode='human')
env = RelativePosition(env)     # refer to gym_examples/wrappers/relative_position.py, observation space has changed!


# ####################  hyper parameters  ####################
qnet_type = 'MLP'
number_timesteps = 10000  # total number of time steps to train on
test_number_timesteps = 1000    # in test phase
explore_timesteps = number_timesteps
# epsilon-greedy schedule, final exploit prob is 0.99
epsilon = lambda i_iter: (1 - 0.99 * min(1, i_iter / explore_timesteps)) * 0.8
lr = 5e-3  # learning rate
buffer_size = explore_timesteps//10*200  # replay buffer size
target_q_update_freq = 100  # how frequency target q net update
ob_scale = 1.0  # scale observations
clipnorm = None


in_dim = env.observation_space.shape
out_dim = env.action_space.n
reward_gamma = 0.99  # reward discount
batch_size = 128  # batch size for sampling from replay buffer
warm_start = batch_size*2  # sample times befor learning
noise_update_freq = 50 # how frequency param noise net update


# ##############################  Network  ####################################
class MLP(tl.models.Model):

    def __init__(self, name):
        super(MLP, self).__init__(name=name)
        hidden_dim = 256
        self.h1 = tl.layers.Dense(hidden_dim, tf.nn.tanh, in_channels=in_dim[0])
        self.qvalue = tl.layers.Dense(out_dim, in_channels=hidden_dim, name='q', W_init=tf.initializers.GlorotUniform(), b_init=tf.constant_initializer(0.1))
        self.svalue = tl.layers.Dense(1, in_channels=hidden_dim, name='s', W_init=tf.initializers.GlorotUniform(), b_init=tf.constant_initializer(0.1))
        self.noise_scale = 0

    def forward(self, ni):
        feature = self.h1(ni)

        # apply noise to all linear layer
        if self.noise_scale != 0:
            noises = []
            for layer in [self.qvalue, self.svalue]:
                for var in layer.trainable_weights:
                    noise = tf.random.normal(tf.shape(var), 0, self.noise_scale)
                    noises.append(noise)
                    var.assign_add(noise)

        qvalue = self.qvalue(feature)
        svalue = self.svalue(feature)

        if self.noise_scale != 0:
            idx = 0
            for layer in [self.qvalue, self.svalue]:
                for var in layer.trainable_weights:
                    var.assign_sub(noises[idx])
                    idx += 1

        if dueling:
            # dueling network
            return svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)
        else:
            return qvalue


class CNN(tl.models.Model):

    def __init__(self, name):
        super(CNN, self).__init__(name=name)
        h, w, in_channels = in_dim
        dense_in_channels = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.conv1 = tl.layers.Conv2d(
            32, (8, 8), (4, 4), tf.nn.relu, 'VALID', in_channels=in_channels, name='conv2d_1',
            W_init=tf.initializers.GlorotUniform()
        )
        self.conv2 = tl.layers.Conv2d(
            64, (4, 4), (2, 2), tf.nn.relu, 'VALID', in_channels=32, name='conv2d_2',
            W_init=tf.initializers.GlorotUniform()
        )
        self.conv3 = tl.layers.Conv2d(
            64, (3, 3), (1, 1), tf.nn.relu, 'VALID', in_channels=64, name='conv2d_3',
            W_init=tf.initializers.GlorotUniform()
        )
        self.flatten = tl.layers.Flatten(name='flatten')
        self.preq = tl.layers.Dense(
            256, tf.nn.relu, in_channels=dense_in_channels, name='pre_q', W_init=tf.initializers.GlorotUniform()
        )
        self.qvalue = tl.layers.Dense(out_dim, in_channels=256, name='q', W_init=tf.initializers.GlorotUniform())
        self.pres = tl.layers.Dense(
            256, tf.nn.relu, in_channels=dense_in_channels, name='pre_s', W_init=tf.initializers.GlorotUniform()
        )
        self.svalue = tl.layers.Dense(1, in_channels=256, name='state', W_init=tf.initializers.GlorotUniform())
        self.noise_scale = 0

    def forward(self, ni):
        feature = self.flatten(self.conv3(self.conv2(self.conv1(ni))))

        # apply noise to all linear layer
        if self.noise_scale != 0:
            noises = []
            for layer in [self.preq, self.qvalue, self.pres, self.svalue]:
                for var in layer.trainable_weights:
                    noise = tf.random.normal(tf.shape(var), 0, self.noise_scale)
                    noises.append(noise)
                    var.assign_add(noise)

        qvalue = self.qvalue(self.preq(feature))
        svalue = self.svalue(self.pres(feature))

        if self.noise_scale != 0:
            idx = 0
            for layer in [self.preq, self.qvalue, self.pres, self.svalue]:
                for var in layer.trainable_weights:
                    var.assign_sub(noises[idx])
                    idx += 1

        if dueling:
            # dueling network
            return svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)
        else:
            return qvalue


# ##############################  Original Replay Buffer  ####################################
class ReplayBuffer(object):

    def __init__(self, size):
        self._storage = []   #保存的容器
        self._maxsize = size #容器最大的size
        self._next_idx = 0   #指针，表示当前新增位置


    #查询这个容器的大小
    def __len__(self):
        return len(self._storage)

    #把信息放入buffer
    def add(self, *args):
        r = args[2]
        #如果当前指针大于容器目前大小，那么扩展容器，append数据
        if self._next_idx >= len(self._storage):
            self._storage.append(args)
        #如果不是，直接写进去就可以了。
        else:
            self._storage[self._next_idx] = args
        #这是一个循环指针
        self._next_idx = (self._next_idx + 1) % self._maxsize

    #对
    def _encode_sample(self, idxes):
        b_o, b_a, b_r, b_o_, b_d = [], [], [], [], []
        for i in idxes:
            o, a, r, o_, d = self._storage[i]
            b_o.append(o)
            b_a.append(a)
            b_r.append(r)
            b_o_.append(o_)
            b_d.append(d)
        return (
            np.stack(b_o).astype('float32') * ob_scale,
            np.stack(b_a).astype('int32'),
            np.stack(b_r).astype('float32'),
            np.stack(b_o_).astype('float32') * ob_scale,
            np.stack(b_d).astype('float32'),
        )

    #抽取数据
    def sample(self, batch_size):
        indexes = [i for i in range(len(self._storage))]
        idxes = [random.choice(indexes) for _ in range(batch_size)]
        return self._encode_sample(idxes)


# #############################  Functions  ###################################
def huber_loss(x):
    """Loss function for value"""
    return tf.where(tf.abs(x) < 1, tf.square(x) * 0.5, tf.abs(x) - 0.5)


def sync(net, net_tar):
    """Copy q network to target q network"""
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)


def log_softmax(x, dim):
    temp = x - np.max(x, dim, keepdims=True)
    return temp - np.log(np.exp(temp).sum(dim, keepdims=True))


def softmax(x, dim):
    temp = np.exp(x - np.max(x, dim, keepdims=True))
    return temp / temp.sum(dim, keepdims=True)




##################### DQN with PER ##########################

class DQN(object):

    def __init__(self):
        model = MLP if qnet_type == 'MLP' else CNN
        self.qnet = model('q')
        if args.mode == 'train':
            self.qnet.train()
            self.targetqnet = model('targetq')
            self.targetqnet.infer()
            sync(self.qnet, self.targetqnet)
        else:
            self.qnet.infer()
            print("Begin loading ... \n\n")
            tl.files.load_and_assign_npz_dict(name=args.save_path, network=self.qnet)
            print("Successfully loaded ... \n\n")
        self.niter = 0
        if clipnorm is not None:
            self.optimizer = tf.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)
        else:
            self.optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.noise_scale = noise_scale

        self.memory = Memory(buffer_size)

    def get_action(self, obv):
        '''
            @Action -- 0 right, 1 up, 2 left, 3 down
            @Observation -- {[x1-x2, y1-y2]}, agent_loc and target_loc
        '''
        eps = epsilon(self.niter)
        if args.mode == 'train':
            if random.random() < eps:       
                if random.random() < 0.8:   
                    return int(random.random() * out_dim)
                return self._get_action_from_APF(obv)   # APF
            obv = np.expand_dims(obv, 0).astype('float32') * ob_scale
            if self.niter < explore_timesteps:
                self.qnet.noise_scale = self.noise_scale
                q_ptb = self._qvalues_func(obv).numpy()
                self.qnet.noise_scale = 0
                if i % noise_update_freq == 0:
                    q = self._qvalues_func(obv).numpy()
                    kl_ptb = (log_softmax(q, 1) - log_softmax(q_ptb, 1))
                    kl_ptb = np.sum(kl_ptb * softmax(q, 1), 1).mean()
                    kl_explore = -np.log(1 - eps + eps / out_dim)
                    if kl_ptb < kl_explore:
                        self.noise_scale *= 1.01
                    else:
                        self.noise_scale /= 1.01
                return q_ptb.argmax(1)[0]
            else:
                return self._qvalues_func(obv).numpy().argmax(1)[0]
        else:
            # Test phase
            if random.random() < 0.1:          
                return self._get_action_from_APF(obv)   # APF
            obv = np.expand_dims(obv, 0).astype('float32') * ob_scale
            return self._qvalues_func(obv).numpy().argmax(1)[0]

    def _get_action_from_APF(self, obv):
        '''
            obv: shape (2 + 25*6)
        '''
        attraction_force = obv[:2]   # shape (2,)
        attraction_force = np.where(attraction_force !=0, 1/attraction_force, 0)    # shape(2,)

        repulsion_force = np.array([0.0,0.0])
        for i in range(25):
            idx = 2 + i*6
            if obv[idx+4] == 1:  # [x,y, 是否可行、目标、障碍物、越界]
                # 观察到障碍物
                obs_pos = obv[idx:(idx+2)]
                repulsion_force += np.where(obs_pos !=0, 1/obs_pos, 0)

        force = attraction_force - repulsion_force
        # 根据合力方向选择移动方向
        x,y = force[0], force[1]
        action = 0
        if y>=x:
            if y>=-x: action = 3  # down
            else: action = 2  # left
        else:
            if y>=-x: action = 0  # right
            else: action = 1  # up
        return action



    @tf.function
    def _qvalues_func(self, obv):
        return self.qnet(obv)

    def append_sample(self, state, action, reward, next_state, done):
        target = self.qnet(tf.constant(state[None], dtype=tf.float32)).numpy()
        old_val = target[0][action]
        target_val = self.targetqnet(tf.constant(next_state[None], dtype=tf.float32)).numpy()
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + reward_gamma * np.max(target_val)

        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, 1 if done else 0))   # save samples


    def train(self):
        mini_batch, idxs, is_weights = self.memory.sample(batch_size)

        b_o = np.array([mini_batch[i][0] for i in range(batch_size)], dtype=np.float32)
        b_a = np.array([mini_batch[i][1] for i in range(batch_size)])
        b_r = np.array([mini_batch[i][2] for i in range(batch_size)], dtype=np.float32)
        b_o_ = np.array([mini_batch[i][3] for i in range(batch_size)], dtype=np.float32)
        b_d = np.array([mini_batch[i][4] for i in range(batch_size)], dtype=np.float32)
        is_weights = np.array(is_weights, dtype=np.float32)

        tf_errors = self._train_func(b_o, b_a, b_r, b_o_, b_d, is_weights)

        # update priority
        for i in range(batch_size):
            idx = idxs[i]
            self.memory.update(idx, tf_errors[i])


        self.niter += 1
        if self.niter % target_q_update_freq == 0:
            sync(self.qnet, self.targetqnet)
        if self.niter % (500) == 0:
            path = os.path.join(args.save_path, '{}.npz'.format(self.niter))
            n = self.qnet.trainable_weights
            print('Save model!\n\n\n\n\n')
            tl.files.save_npz_dict(self.qnet.trainable_weights, name=path)

    @tf.function  
    def _train_func(self, b_o, b_a, b_r, b_o_, b_d, is_weights):
        with tf.GradientTape() as tape:
            td_errors = self._tderror_func(b_o, b_a, b_r, b_o_, b_d) 
            loss = is_weights * tf.reduce_mean(huber_loss(td_errors))

        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))

        return td_errors

    @tf.function
    def _tderror_func(self, b_o, b_a, b_r, b_o_, b_d):
        if double:
            b_a_ = tf.one_hot(tf.argmax(self.qnet(b_o_), 1), out_dim)
            b_q_ = (1 - b_d) * tf.reduce_sum(self.targetqnet(b_o_) * b_a_, 1)
        else:
            b_q_ = (1 - b_d) * tf.reduce_max(self.targetqnet(b_o_), 1)

        b_q = tf.reduce_sum(self.qnet(b_o) * tf.one_hot(b_a, out_dim), 1)
        return b_q - (b_r + reward_gamma * b_q_)





if __name__=='__main__':
    dqn = DQN()
    if args.mode == 'train':

        o,_ = env.reset(seed=args.seed)         # 由于seed，每次起始点终点一样
        nepisode = 0
        t = time.time()
        for i in range(1, number_timesteps + 1):

            a = dqn.get_action(o)

            # execute action and feed to replay buffer
            # note that `_` tail in var name means next
            o_, r, done, _, info = env.step(a)
            dqn.append_sample(o, a, r, o_, done)

            if dqn.memory.tree.n_entries >= warm_start:
                dqn.train()

            if done:
                o,_ = env.reset()
                nepisode += 1
            else:
                o = o_

            # episode in info is real (unwrapped) message
            if info.get('episode'):
                reward, length = info['episode']['r'], info['episode']['l']
                fps = int(length / (time.time() - t))
                print(
                    'Time steps so far: {}, episode so far: {}, '
                    'episode reward: {:.4f}, episode length: {}, FPS: {}'.format(i, nepisode, reward, length, fps)
                )
                t = time.time()

    #  Test phase
    else:
        nepisode = 0    # completed episodes, including episodes that exceeding the maximum step length and 'suc_epi'
        suc_epi = 0     # episodes that agent acheives the target.
        o,_ = env.reset()
        for i in tqdm(range(1, test_number_timesteps + 1)):
            a = dqn.get_action(o)

            # execute action
            # note that `_` tail in var name means next
            # print("1\n\n\n")
            o_, r, done, _, info = env.step(a)

            if done:
                if info['episode']['achieve_target']:
                    suc_epi += 1
                o,_ = env.reset()
                nepisode += 1
            else:
                o = o_

            # episode in info is real (unwrapped) message
            if info.get('episode'):
                reward, length = info['episode']['r'], info['episode']['l']
                print(
                    'Time steps so far: {}, episode so far: {}, '
                    'episode reward: {:.4f}, episode length: {}'.format(i, nepisode, reward, length)
                )

        print(f'Successful episode: {suc_epi} / {nepisode}')        