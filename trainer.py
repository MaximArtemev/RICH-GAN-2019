from torch.optim import AdamW

from tqdm import tqdm

from model import *

class Trainer:
    def __init__(
        self
        , train_loader
        , val_loader
        , noise_size
        , hidden_size
        , num_layers
        , cramer_size
        , epochs=1
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.generator = Generator(
            noise_size
            , hidden_size
            , num_layers
        ).to(device)

        self.critic = Critic(
            hidden_size
            , cramer_size
            , num_layers
        ).to(device)

        self.g_optimizer = AdamW(self.generator.parameters())
        self.c_optimizer = AdamW(self.critic.parameters())

        self.epochs = epochs

    def train(self, print_every=100):
        for epoch in range(1, self.epochs+1):
            print(f"(epoch {epoch})")

            #iter = 0
            g_avg_loss = 0
            c_avg_loss = 0
            for (real_1, noised_1, noised_2, w_1, w_2) in tqdm(self.train_loader):
                #iter += 1

                self.g_optimizer.zero_grad()
                self.c_optimizer.zero_grad()

                w_1 = w_1.unsqueeze(1)
                w_2 = w_2.unsqueeze(1)

                generated_1 = self.generator(noised_1)
                generated_2 = self.generator(noised_2)

                verdict_1 = self.critic(real_1, generated_2)
                verdict_2 = self.critic(generated_1, generated_2)

                # TODO штрафы за градиент

                g_loss = ((verdict_1 - verdict_2) * w_1 * w_2).mean()
                g_loss.backward(retain_graph=True)
                self.g_optimizer.step()

                c_loss = -g_loss
                c_loss.backward()
                self.c_optimizer.step()

                c_avg_loss += c_loss.item()

                # if iter % print_every == 0:
                #     print(c_avg_loss/print_every)
                #     c_avg_loss = 0
