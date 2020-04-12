import numpy as np

from torch.optim import AdamW

from tqdm import tqdm

from teacher_model import *

class TeacherTrainer:
    def __init__(
        self
        , train_loader
        , val_loader
        , noise_size
        , hidden_size
        , num_layers
        , cramer_size
        , lam
        , epochs
        , critic_boost
        , lr
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.generator = TeacherGenerator(
            noise_size
            , hidden_size
            , num_layers
        ).to(device)

        self.critic = TeacherCritic(
            hidden_size
            , cramer_size
            , num_layers
        ).to(device)

        self.g_optimizer = AdamW(self.generator.parameters(), lr=lr)
        self.c_optimizer = AdamW(self.critic.parameters(), lr=lr)

        self.lam = lam
        self.epochs = epochs
        self.critic_boost = critic_boost

    def train(self):
        for epoch in range(1, self.epochs+1):
            print(f"(epoch {epoch})")

            g_avg_loss = 0
            c_avg_loss = 0


            step = 0
            for (real, noised_1, noised_2, w_real, w_1, w_2) in tqdm(self.train_loader):
                step += 1

                gen_1= self.generator(noised_1)
                gen_2 = self.generator(noised_2)

                g_loss = (
                    (
                        self.critic(real, gen_2) - self.critic(gen_1, gen_2)
                    ) * w_1.unsqueeze(1) * w_2.unsqueeze(1)
                ).mean()
                self.g_optimizer.zero_grad()
                if step % self.critic_boost == 0:
                    g_loss.backward(retain_graph=True)
                    self.g_optimizer.step()


                alpha = torch.tensor(np.random.uniform(low=0.0, high=1.0, size=real.size(0))).unsqueeze(1).float().to(device)
                blended = real * alpha + gen_1 * (1 - alpha)
                f = self.critic(blended, gen_2).mean()
                blend_grad = torch.autograd.grad(f, blended)[0]

                self.c_optimizer.zero_grad()
                c_loss = -g_loss + self.lam * ((blend_grad.norm(dim=1) - 1) ** 2).mean()
                c_loss.backward()
                self.c_optimizer.step()

            real_batches = []
            gen_batches = []
            w_batches = []

            #with torch.no_grad():
            for (real, noised, w) in tqdm(self.val_loader):
                gen = self.generator(noised)

                real_batches.append(real.detach().cpu())
                gen_batches.append(gen.detach().cpu())
                w_batches.append(w.detach().cpu())

            plot_distributions(
                torch.cat(real_batches, dim=0)
                , torch.cat(gen_batches, dim=0)
                , torch.cat(w_batches, dim=0)
            )
