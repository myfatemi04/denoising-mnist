# Coding along with nn.labml.ai/diffusion/ddpm/unet.html.

import math
from matplotlib import pyplot as plt
import tqdm

import torch
import torch.nn as nn
import numpy as np
import random

from denoising_unet import UNet

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def visualize_noise_levels(sample, sqrt_alphabar, sqrt_one_minus_alphabar):
    plt.rcParams['figure.figsize'] = [12, 4]

    num_timesteps = len(sqrt_alphabar)

    denormalize = lambda sample: ((sample + 1) / 2).permute(1, 2, 0).detach().cpu().numpy()

    nshown = 11
    counter = 0
    for timestep in range(0, num_timesteps + 1):
        if timestep == 0:
            plt.subplot(1, nshown, 1)
            plt.imshow(denormalize(sample), cmap='gray')
            plt.axis('off')
            plt.title('Original')
        else:
            noise = torch.randn_like(sample)
            sample_noisy = sample * sqrt_alphabar[timestep - 1] + noise * sqrt_one_minus_alphabar[timestep - 1]

            # t = torch.tensor([timestep], dtype=torch.float)

            # time_embeddings = time_embedder(t)
            # noise = model(sample.unsqueeze(0), time_embeddings)
            # noise = noise[0].detach().cpu().numpy()

            plt.subplot(1, nshown, timestep + 1)
            plt.imshow(denormalize(sample_noisy), cmap='gray')
            plt.axis('off')
            plt.title(f't={timestep}')

        counter += 1

    plt.tight_layout()
    plt.show()
            

def main():
    # Let's train our model! Iteration 01: No classifier guidance.
    # Note that attention is still used even without classifier guidance.
    # In the example code, attention is only used in the later layers,
    # where presumably the feature vectors are much richer.
    import sklearn.datasets
    import numpy as np

    ### Data ###
    # Load the MNIST dataset.
    mnist = sklearn.datasets.fetch_openml('mnist_784', version=1)

    X_df = mnist.data
    X_numpy = X_df.to_numpy().astype(float)
    X = torch.tensor(X_numpy / 255.0).view(-1, 1, 28, 28)
    # Pad to size 32
    X = torch.nn.functional.pad(X, (2, 2, 2, 2), value=0)

    y_df = mnist.target
    y = torch.tensor(y_df.to_numpy().astype(float))

    ### Noise Scheduler ###
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_levels = 10
    beta_min = 4e-4
    beta_max = 0.02
    betas = -(torch.cos(torch.linspace(0, np.pi, noise_levels, device=device)) - 1) / 2 * (beta_max - beta_min) + beta_min
    alphabar = torch.cumprod(1 - betas, dim=0)
    # scale to make alphabar = 0 at the end
    orig_alphabar0 = alphabar[0]
    alphabar = alphabar - alphabar[-1]
    alphabar = alphabar * orig_alphabar0 / alphabar[0]
    # signal coefficients
    sqrt_alphabar = torch.sqrt(alphabar)
    # noise std. deviations
    sqrt_one_minus_alphabar = torch.sqrt(1 - alphabar)
    sigmas = sqrt_one_minus_alphabar

    ### Visualize Noise ###
    if False:
        plt.plot(betas)
        plt.title("$\\beta$ values")
        plt.show()

        plt.plot(sqrt_alphabar)
        plt.title("$\\sqrt{\\overline{\\alpha}}$ values")
        plt.show()

        visualize_noise_levels((X[0] + 1) / 2, sqrt_alphabar, sqrt_one_minus_alphabar)

        exit()

    ### Train Model ###
    model = UNet(
        image_channels=1,
        n_channels=16,
        channel_multiples=[1, 2, 2, 4],
        is_attention=[False, False, True, True],
        n_blocks=2
    ).to(device)
    X = X.to(device).float()
    y = y.to(device)
    # X = X[y == 5] # specifically only train on the number "5"
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    width, height = 32, 32
    batch_size = 512

    loss_hist = []
    error = []

    ### !!! IMPORTANT PARAMETER !!! ###
    do_baseline = False

    counter = 0
    for epoch in range(100 if not do_baseline else 1):
        for i in (pbar := tqdm.tqdm(range(0, len(X), batch_size), desc='Training epoch %d' % epoch)):
            x = X[i:i + batch_size]
            x_normalized = x * 2 - 1
            t = torch.randint(1, noise_levels + 1, (x.shape[0], 1, 1, 1), device='cuda')
            noise = torch.randn_like(x)
            x_noisy = x_normalized * sqrt_alphabar[t - 1] + noise * sqrt_one_minus_alphabar[t - 1]

            if do_baseline:
                noise_pred = torch.zeros_like(x_noisy)
            else:
                noise_pred = model(x_noisy, t.view(x.shape[0]).float())
            # this is the error per item
            squared_error = ((noise_pred - noise) ** 2).mean(dim=(1, 2, 3))
            # scale error for each instance based on the timestep
            loss = 1/2 * (squared_error * (sigmas[t - 1] ** -2)).mean()

            # visualize error as a function of timestep
            error.append((squared_error, t))

            if torch.isnan(loss):
                print("Loss is NaN.")

                # Go through the model again, but this time tracing for NaNs.
                # Check for NaNs in model parameters
                num_nan_parameters = 0
                for name, param in model.named_parameters():
                    nan_mask = torch.isnan(param.data)
                    if torch.any(nan_mask):
                        num_nan_parameters += 1
                        # print(f"{nan_mask.nonzero().numel()} NaN values found in parameter '{name}'")
                print("Number of NaN parameters:", num_nan_parameters, "out of", sum(1 for _ in model.named_parameters()))
                print(counter)

                exit()
            
            if not do_baseline:
                optim.zero_grad()
                loss.backward()
                # gradient clipping
                hasnan = False
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print("Param", name, "has no gradient")
                    elif torch.isnan(param.grad).any():
                        print("Param", name, "has NaN element in gradient")
                        hasnan = True
                if hasnan:
                    print("loss was", loss)
                    exit()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            pbar.set_postfix(loss=loss.item())
            loss_hist.append(loss.item())

            counter += 1

    if not do_baseline:
        torch.save(model.state_dict(), 'mnist_generator.pth')
        np.save('mnist_generator_loss.npy', np.array(loss_hist))
        import pickle
        with open('mnist_generator_error.pkl', 'wb') as f:
            pickle.dump(error, f)
    else:
        import pickle
        with open('mnist_generator_error_baseline.pkl', 'wb') as f:
            pickle.dump(error, f)

if __name__ == '__main__':
    main()
