from tqdm import tqdm as tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter


class AC_Trainer:
    def __init__(self, vae_model, actor, real_critic, attr_critic, num_epochs,
                 trainDataLoader, valDataLoader, device=0, lr=1e-4):
        self.vae_model = vae_model
        self.actor = actor
        self.real_critic = real_critic
        self.attr_critic = attr_critic

        self.start_epoch = 1

        self.n_train = len(trainDataLoader.dataset)
        self.n_val = len(valDataLoader.dataset)

        self.device = device

        self.n_latent = vae_model.latent_size

        self.iteration = 0

        self.num_epochs = num_epochs
        self.trainDataLoader = trainDataLoader

        self.real_optim = getattr(optim, 'Adam')(self.real_critic.parameters(), lr=lr * 3)
        self.actor_optim = getattr(optim, 'Adam')(self.actor.parameters(), lr=lr * 3)

        self.tensorboad_writer = SummaryWriter()

        self.percentage_prior_fake = 0.1
        self.N_between_update_G = 10
        self.N_between_eval = 100

    def train(self):
        for epoch in tqdm(range(self.start_epoch, self.num_epochs+1)):
            self.train_epoch(epoch)
            # TODO: Get sample

            if epoch % 10:
                self.save_model(epoch)

            # Decay learning rates
            if (epoch + 1) == 30:
                self.actor_optim.param_groups[0]['lr'] /= 10
                self.real_optim.param_groups[0]['lr'] /= 10
            if (epoch + 1) == 50:
                self.actor_optim.param_groups[0]['lr'] /= 10
                self.real_optim.param_groups[0]['lr'] /= 10

        print("[+] Finished Training Model")

    def train_epoch(self, epoch):
        self.vae_model.eval()
        self.actor.train()
        self.real_critic.train()
        self.attr_critic.train()

        # Traces
        train_loss, iteration, total_actor_loss, total_real_loss, total_dist_penalty, actor_iteration = 0, 0, 0, 0, 0, 0

        for batch_idx, batch in enumerate(self.trainDataLoader):
            batch_size = batch['input'].size(0)

            labels = batch['phrase_tags']
            labels = labels.to(self.device)

            self.iteration += 1
            iteration += 1

            real_data = torch.ones(batch_size, 1).to(self.device)
            fake_data = torch.zeros(batch_size, 1).to(self.device)
            fake_z_prior = torch.randn(batch_size, self.n_latent).to(self.device)
            fake_attributes = self.get_fake_attributes(batch_size).to(self.device)

            with torch.no_grad():
                _, _, logv, real_z = self.vae_model(batch['input'], batch['length'])

            fake_z_prior.requires_grad = True
            real_z.requires_grad = True
            labels.requires_grad = True
            fake_attributes.requires_grad = True

            self.real_critic.zero_grad()

            # We train D by sampling from p(z) at a rate 10 times less than G(p(z))
            if np.random.randn(1) < self.percentage_prior_fake:
                # Use Prior for fake_samples
                input_data = torch.cat([real_z, fake_z_prior, real_z], dim=0)

            else:
                # Use Generator to make fake_samples
                fake_z_gen = self.actor(fake_z_prior, labels)
                input_data = torch.cat([real_z, fake_z_gen, real_z], dim=0)

            input_attr = torch.cat([labels, labels, fake_attributes], dim=0)
            real_labels = torch.cat([real_data, fake_data, fake_data])
            logit_out = self.real_critic(input_data, input_attr)
            critic_loss = F.binary_cross_entropy(logit_out, real_labels)

            critic_loss.backward()
            self.real_optim.step()

            if (batch_idx + 1) % 1000 == 0:
                self.d_critic_histogram(self.iteration)

            total_real_loss += critic_loss.item()

            # Train actor
            if (batch_idx + 1) % self.N_between_update_G == 0:
                self.actor.zero_grad()
                fake_z_prior = torch.randn(batch_size, self.n_latent).to(self.device)
                fake_z_prior = self.re_allocate(fake_z_prior)

                actor_labels = labels
                actor_truth = real_data

                actor_labels.requires_grad = True
                fake_z_prior.requires_grad = True

                actor_g = self.actor(fake_z_prior, actor_labels)
                real_g = self.actor(real_z, actor_labels)

                zg_critic_out = self.real_critic(actor_g, actor_labels)
                zg_critic_real = self.real_critic(real_g, actor_labels)

                weight_var = torch.mean(logv, 0, True)  # TODO: might have to use sigma**2 instead of logv
                dist_penalty = torch.mean(
                    torch.sum((1 + (actor_g - fake_z_prior).pow(2)).log() * weight_var.pow(-2), 1), 0)
                dist_penalty = dist_penalty + torch.mean(
                    torch.sum((1 + (real_g - real_z).pow(2)).log() * weight_var.pow(-2), 1), 0)

                actor_loss = F.binary_cross_entropy(zg_critic_out, actor_truth,
                                                    size_average=False) + F.binary_cross_entropy(zg_critic_real,
                                                                                                 actor_truth,
                                                                                                 size_average=False) + dist_penalty

                actor_loss.backward()
                total_actor_loss += actor_loss.item()
                self.actor_optim.step()
                total_dist_penalty += dist_penalty.item()
                actor_iteration += 1
                if (actor_iteration % 100) == 0:
                    self.d_actor_histogram(self.iteration)

        print("Distance penalty: {} , {} | Critic loss: {}".format(total_dist_penalty / actor_iteration,
                                                                   total_actor_loss / actor_iteration,
                                                                   total_real_loss / iteration))

        self.summary_write(total_dist_penalty / actor_iteration, total_actor_loss / actor_iteration,
                            total_real_loss / iteration, epoch)
        print("[+] Epoch:[{}/{}] train actor average loss :{}".format(epoch, self.num_epochs, train_loss))

    def re_allocate(self, data):
        new_data = data.detach()
        new_data.requires_grad = True
        return new_data

    def d_critic_histogram(self, iteration):
            for name, param in self.real_critic.named_parameters():  # actor
                self.tensorboad_writer.add_histogram('real_critic/' + name, param.clone().cpu().data.numpy(), iteration,
                                                     bins='sturges')
                self.tensorboad_writer.add_histogram('real_critic/' + name + '/grad',
                                                     param.grad.clone().cpu().data.numpy(), iteration, bins='sturges')

    def d_actor_histogram(self, iteration):
        for name, param in self.actor.named_parameters():  # actor
            self.tensorboad_writer.add_histogram('actor/' + name, param.clone().cpu().data.numpy(), iteration,
                                                 bins='sturges')
            self.tensorboad_writer.add_histogram('actor/' + name + '/grad', param.grad.clone().cpu().data.numpy(),
                                                 iteration, bins='sturges')

    def get_fake_attributes(self, batch_size, num_labels=10):
        start = np.random.randint(self.n_train - batch_size - 1)
        data = self.trainDataLoader.dataset[start:start + batch_size] # TODO: how to index this, damnit jSON!
        fake_attributes = []

        for (name, labels) in data:  # TODO: update this when Joseph does the label thing
            fake_attributes.append(labels)

        return torch.FloatTensor(fake_attributes)

    def _set_label_type(self):

        return

    def summary_write(self, distance_penalty, loss, real_loss, epoch):
        self.tensorboad_writer.add_scalar('data/loss', loss, epoch)  # need to modify . We use four loss value .
        self.tensorboad_writer.add_scalar('data/distance_penalty', distance_penalty,
                                          epoch)  # need to modify . We use four loss value .
        self.tensorboad_writer.add_scalar('data/discriminator_loss', real_loss, epoch)

    def save_model(self, epoch):
        torch.save(self.actor.state_dict(), './save_model/actor_model' + str(epoch) + '.path.tar')
        torch.save(self.real_critic.state_dict(),
                   './save_model/real_d_model' + str(epoch) + '.path.tar')
        torch.save(self.attr_critic.state_dict(),
                   './save_model/attr_d_model' + str(epoch) + '.path.tar')
