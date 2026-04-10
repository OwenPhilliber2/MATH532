from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

show_img = False
epsilon = 1e-8

img = Image.open('Homework2/data/CandO.JPEG')
img_arr = np.rot90(np.array(img), k = 3)


if show_img:
    plt.imshow(img_arr)
    plt.show()

red_channel = img_arr[:, :, 0]
green_channel = img_arr[:, :, 1]
blue_channel = img_arr[:, :, 2]

U_r, S_r, Vh_r = np.linalg.svd(red_channel, full_matrices=False)
U_g, S_g, Vh_g = np.linalg.svd(green_channel, full_matrices=False)
U_b, S_b, Vh_b = np.linalg.svd(blue_channel, full_matrices=False)

print(S_r.shape)

fig, axs = plt.subplots(3)
fig.suptitle('Singular Values')
axs[0].plot(S_r, 'tab:red')
axs[0].set_yscale('log')
axs[1].plot(S_g, 'tab:green')
axs[1].set_yscale('log')
axs[2].plot(S_b, 'tab:blue')
axs[2].set_yscale('log')
plt.show()

def outer_prod(index):
    channel_red = S_r[index] * np.outer(U_r[:, index], Vh_r[index, :])
    channel_green = S_g[index] * np.outer(U_g[:, index], Vh_g[index, :])
    channel_blue = S_b[index] * np.outer(U_b[:, index], Vh_b[index, :])

    return_img = np.stack((channel_red, channel_green, channel_blue), axis=2)
    return np.clip(return_img, 0, 255).astype(np.uint8)

fig, axs = plt.subplots(2, 2)
fig.suptitle('Singular Value Rank One Outer Products')

axs[0,0].imshow(outer_prod(0))
axs[0,0].set_title('k = 1')
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])

axs[0,1].imshow(outer_prod(1))
axs[0,1].set_title('k = 2')
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])

axs[1,0].imshow(outer_prod(2))
axs[1,0].set_title('k = 3')
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])

axs[1,1].imshow(outer_prod(3))
axs[1,1].set_title('k = 4')
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])

plt.tight_layout()
plt.show()

def SVD_approx(k):
    Ar = (U_r[:, :k] * S_r[:k]) @ Vh_r[:k, :]
    Ag = (U_g[:, :k] * S_g[:k]) @ Vh_g[:k, :]
    Ab = (U_b[:, :k] * S_b[:k]) @ Vh_b[:k, :]

    A = np.stack((Ar, Ag, Ab), axis=2)
    return np.clip(A, 0, 255).astype(np.uint8)

fig, axs = plt.subplots(2,2)
fig.suptitle('SVD Approximations')
axs[0,0].imshow(SVD_approx(10))
axs[0,0].set_title('k = 10')
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,1].imshow(SVD_approx(20))
axs[0,1].set_title('k = 20')
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[1,0].imshow(SVD_approx(30))
axs[1,0].set_title('k = 30')
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])
axs[1,1].imshow(SVD_approx(40))
axs[1,1].set_title('k = 40')
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])
plt.show()


fig, axs = plt.subplots(2,2)
fig.suptitle('SVD Approximations')
axs[0,0].imshow(SVD_approx(1))
axs[0,0].set_title('k = 1')
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])
axs[0,1].imshow(SVD_approx(2))
axs[0,1].set_title('k = 2')
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[1,0].imshow(SVD_approx(3))
axs[1,0].set_title('k = 3')
axs[1,0].set_xticks([])
axs[1,0].set_yticks([])
axs[1,1].imshow(SVD_approx(4))
axs[1,1].set_title('k = 4')
axs[1,1].set_xticks([])
axs[1,1].set_yticks([])
plt.show()


error_vec = np.log(np.abs(img_arr - SVD_approx(100)) + epsilon)

print(np.min(error_vec), np.max(error_vec), type(error_vec))

plt.imshow(error_vec)
plt.xticks([])
plt.yticks([])
plt.title("Error Plot for k = 100")
plt.show()