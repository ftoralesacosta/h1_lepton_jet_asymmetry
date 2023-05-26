from PIL import Image
from PIL import ImageDraw,ImageFont
import glob
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='./', help='Folder containing figures')
parser.add_argument('--niter', type=int, default=4, help='Number of iterations to run over')
flags = parser.parse_args()


# Create the frames

base_folder = "."
#to_gif = flags.plot
# font = ImageFont.truetype("./Helvetica-Bold.ttf", size=35)

plot_list = {
    "Phi_Distributions",
    "CosPhi_Distributions"
}

for to_gif in plot_list:
    frames = []
    for i in range(0,flags.niter):
        print(os.path.join(base_folder,"{}_Iteration_{}.png".format(to_gif,i)))
        new_frame = Image.open(os.path.join(base_folder,"{}_Iteration_{}.png".format(to_gif,i)))
        # draw = ImageDraw.Draw(new_frame)
        # draw.text((120, 75), "Iteration {}".format(i),fill="black",font=font)
        frames.append(new_frame)
 
    # Save into a GIF file that loops forever
    frames[0].save(os.path.join(base_folder,'{}.gif'.format(to_gif)), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=500, loop=0)
