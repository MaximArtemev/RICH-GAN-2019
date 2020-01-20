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
'''
    @torch.no_grad()
    def test(self, beam_size):
        open("errors.txt", "w", encoding='utf-8').close()
        self.model.eval()
        total_p = 0
        total_l = 0
        total_o = 0

        matches_p = 0
        matches_l = 0
        matches_o = 0

        distance_sum_p = 0
        distance_sum_l = 0
        distance_sum_o = 0

        n = 0
        start = time.time()
        total = len(self.test_loader)
        for (
            front_tensor
            , entity_tensor
            , back_tensor
            , target_tensor
            , front_length
            , entity_length
            , back_length
            , target_length
            , tag
        ) in self.test_loader:
            n += 1

            out = self.model(
                front_tensor
                , entity_tensor
                , back_tensor
                , front_length
                , entity_length
                , back_length
                , beam_size=beam_size
            )

            front = self.i2s([token.item() for token in front_tensor[0]], rev=True)
            entity = self.i2s([token.item() for token in entity_tensor[0]])
            back = self.i2s([token.item() for token in back_tensor[0]])
            target = self.i2s([token.item() for token in target_tensor[0]])
            out = self.i2s(out)

            input = f"{front}[{entity}]{back}"

            metrics = self.metrics(out, target)

            if tag == 0:
                matches_p += metrics[0]
                distance_sum_p += metrics[1]
                total_p += 1
            elif tag == 1:
                matches_l += metrics[0]
                distance_sum_l += metrics[1]
                total_l += 1
            else:
                matches_o += metrics[0]
                distance_sum_o += metrics[1]
                total_o += 1

            if metrics[0] == 0:
                with open("errors.txt", "a", encoding='utf-8') as f:
                    f.write(f"[in]\t{input}\n[out]\t{out}\n[tar]\t{target}\n\n")

            print(f"\rcurrent metrics at {n/total :.3f}: "
                  f" p({total_p}): [{matches_p / total_p if total_p != 0 else 0:.3f}, {distance_sum_p / total_p if total_p != 0 else 0:.3f}]"
                  f" l({total_l}): [{matches_l / total_l if total_l != 0 else 0:.3f}, {distance_sum_l / total_l if total_l != 0 else 0:.3f}]"
                  f" o({total_o}): [{matches_o / total_o if total_o != 0 else 0:.3f}, {distance_sum_o / total_o if total_o != 0 else 0:.3f}]"
                  f" avg: [{(matches_p + matches_l + matches_o) / (total_p + total_l + total_o):.3f}, {(distance_sum_p + distance_sum_l + distance_sum_o) / (total_p + total_l + total_o):.3f}]"
                  f"{time.time() - start :.1f}s", end='')
'''
