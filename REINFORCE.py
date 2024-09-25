import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
# learning_rate = 0.002
gamma         = 0.98

import wandb


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128) # 4 x 128 x 2 네트워크, 4개의 input, 2개의 action
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x): # 네트워크 정의
        x = F.relu(self.fc1(x)) # 활성화함수 relu
        x = F.softmax(self.fc2(x), dim=0) # 분류 softmax
        # softmax를 거치면 확률값이 된다
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        loss = 0
        self.optimizer.zero_grad() # 이전에 계산된 기울기 초기화
        for r, prob in self.data[::-1]: # 데이터를 뒤에서 부터 본다, 맨 뒤는 gamma가 50^이여야하기 때문
            R = r + gamma * R
            loss = -torch.log(prob) * R # log_\pi(s,a) * Reward
            loss.backward() # gradient 계산(backpropagation)
        self.optimizer.step()
        self.data = []


import os

def main():
    epochs = 2000
    wandb.init(
        project="Pangyo_RL_Tuto",
        name=os.path.basename(__file__), # 이 파일이름을 run name으로 지정
        # id="<run-id>",
        config={
        "learning_rate": learning_rate,
        "architecture": "CNN",
        "epochs": epochs,
        }
    )

    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    
    
    for n_epi in range(epochs):
        s, _ = env.reset() 
        # s = [cart pos, cart vel, pole ang, pole vel]
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float()) # action 2개에 대한 확률 분포가 나온다           
            m = Categorical(prob) # Categorical: pytorch에서 지원하는 확률 분포 모델
            a = m.sample() # 상기 확률에 비례해서 action이 나온다
            s_prime, r, done, truncated, info = env.step(a.item()) # tensor의 item으로 값을 넣어줌
            # s_prime: 다음 스탭
            # r: 리워드
            pi.put_data((r,prob[a])) # reinforce 알고리즘은 에피소드가 다 끝나야 학습이 가능하므로 일단 모아둔다
            # prob[a] 은 log_\pi_\theta(s,a) 이다
            s = s_prime
            score += r
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            wandb.log({"avg_score":score/print_interval})
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            # 평균 20스탭동안 평균 몇틱을 버텼다라는 뜻
            score = 0.0
            
    wandb.finish()
    env.close()
    
if __name__ == '__main__':
    main()