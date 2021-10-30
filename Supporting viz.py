# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 19:03:08 2021

@author: Jason
"""
import matplotlib.pyplot as plt

x = np.array(
                [
                    [2.0, 3.0, 1.0, 2.0],
                    [3.0, 2.0, 2.0, 1.0],
                    [3.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 2.0],
                ],
            )

plt.figure(figsize=(4, 5), dpi=d)
x = [[3, 4, 5],
     [2, 3, 4],
     [1, 2, 3]]

color_map = plt.imshow(x)
color_map.set_cmap("Blues_r")
plt.colorbar()