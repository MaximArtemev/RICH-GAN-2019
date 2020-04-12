import numpy as np

from torch.optim import AdamW

from tqdm import tqdm

#from teacher_model import *
from student_model import *

class StudentTrainer:
    def __init__(
        self
        , train_loader
        , val_loader
        # below are the student parameters
        , noise_size
        , student_hidden_size
        , student_num_layers
        , epochs
        , lr
        , teacher_generator
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.teacher_generator = teacher_generator

        self.student_generator = StudentGenerator(
            noise_size
            , student_hidden_size
            , student_num_layers
        ).to(device)

        self.optimizer = AdamW(self.student_generator.parameters(), lr=lr)

        self.epochs = epochs

    def train(self):
        for epoch in range(1, self.epochs+1):
            print(f"(epoch {epoch})")

            avg_loss = 0

            real_batches = []
            teacher_gen_batches = []
            student_gen_batches = []
            w_batches = []
            for (real, noised, w) in tqdm(self.train_loader):

                teacher_generated = self.teacher_generator(noised).detach()
                student_generated = self.student_generator(noised)

                loss = ((teacher_generated - student_generated) ** 2).mean()
                avg_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                real_batches.append(real.detach().cpu())
                teacher_gen_batches.append(teacher_generated.detach().cpu())
                student_gen_batches.append(student_generated.detach().cpu())
                w_batches.append(w.detach().cpu())

            print(f"avg loss: {avg_loss/len(self.train_loader)}")

            plot_three_distributions(
                torch.cat(real_batches, dim=0)
                , torch.cat(teacher_gen_batches, dim=0)
                , torch.cat(student_gen_batches, dim=0)
                , torch.cat(w_batches, dim=0)
            )
