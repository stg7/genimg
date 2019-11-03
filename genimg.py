#!/usr/bin/env python3
import argparse
import sys
import os
import random
import multiprocessing

import skimage.io
import skimage.draw
import sklearn.metrics
import numpy as np


def rmse(x, y):
    return np.sqrt(sklearn.metrics.mean_squared_error(x.flatten(), y.flatten()))


def draw_triangle(i, triangle, target=None, color=None):
    i = i.copy()
    X, Y, Z = triangle
    rr, cc = skimage.draw.polygon(
        np.array([X[0],Y[0],Z[0]]) * (i.shape[0] - 1),
        np.array([X[1],Y[1],Z[1]]) * (i.shape[1] - 1)
    )
    # in case there is no triangle shape there
    if len(rr) == len(cc) == 0:
        return i, None, 0
    if target is not None:
        color = np.median(target[rr,cc], axis=0) #  - np.median(i[rr,cc], axis=0)
    if color is not None:
        i[rr, cc] += color
        #print(color)
        i[rr, cc] /= 2
    area = len(rr)
    return i, color, area


def mutate_triangle(current, triangle, img, mutation_rate):
    t = np.array(triangle) + np.array([(random.uniform(0, 1) - 0.5) * mutation_rate for _ in range(6)]).reshape((3,2))
    t = np.clip(t, 0, 1)
    j, c, area = draw_triangle(current, t, img)
    return (rmse(j, img), t, area)


def check_triangle(current, img, mutation_rate, last_rmse):
    res = []
    triangle = (
        (random.uniform(0,1), random.uniform(0,1)),
        (random.uniform(0,1), random.uniform(0,1)),
        (random.uniform(0,1), random.uniform(0,1))
    )
    j, c, area = draw_triangle(current, triangle, img)
    curr_rmse = rmse(j, img)
    res = [(curr_rmse, triangle, area)]
    last_rmse = min(curr_rmse, last_rmse)

    for i in range(5):
        curr_rmse, t, area = mutate_triangle(current, list(triangle).copy(), img, mutation_rate)
        if curr_rmse < last_rmse:
            res.append((curr_rmse, t, area))
            last_rmse = curr_rmse
    return res




def main(_):
    # argument parsing
    parser = argparse.ArgumentParser(description='image generator',
                                     epilog="stg7 2019",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image", type=str, help="image to process")
    parser.add_argument('--cpu_count', type=int, default=multiprocessing.cpu_count(), help='thread/cpu count')
    parser.add_argument('--num_shapes', type=int, default=200, help='number of used shapes')
    parser.add_argument(
        "--output_file", type=str, default="image.png", help="generated resulting image"
    )
    a = vars(parser.parse_args())
    random.seed(42)
    pool = multiprocessing.Pool(a["cpu_count"])

    num_shapes = a["num_shapes"]

    img = skimage.img_as_float64(
        skimage.io.imread(a["image"])
    )

    aspect_ratio = img.shape[1] / img.shape[0]
    img = skimage.transform.resize(img, (256, int(256 * aspect_ratio)))
    current = np.zeros(img.shape)
    mean_color = img.mean(axis=0).mean(axis=0)

    current[:,:] = mean_color

    high_res = np.zeros((2000, int(2000 * aspect_ratio), 3))
    high_res[:,:] = mean_color
    last_rmse = rmse(current, img)
    mutation_rate = 0.5
    filename = os.path.basename(a["output_file"])
    output_dir = os.path.dirname(a["output_file"])
    # todo: add SVG output
    for j in range(num_shapes):
        candidates = sum(pool.starmap(check_triangle, [(current, img, mutation_rate, last_rmse) for l in range(100)]), [])
        candidates.sort(key=lambda x: (x[0] - last_rmse) * (img.shape[0] * img.shape[1] - x[2] * j/num_shapes))

        last_rmse, triangle, area = candidates[0]
        print(f"step: {j}, rmse: {candidates[0][0]}, len(c)={len(candidates)}, area={area}")
        current, color, _ = draw_triangle(current, triangle, img)
        high_res, _, _ = draw_triangle(high_res, triangle, color=color)

        skimage.io.imsave(output_dir + "/" + str(j) + "_" + filename, skimage.img_as_ubyte(high_res))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

