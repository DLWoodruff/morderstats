"""
morderstats.py
Multivariate Order Statistics
This script should compute the desired prediction regions of a certain type
(Mahalanobis, Halfspace Peeling, Direct Convex Hull Peeling)
and will write and plot all of them in the specified directory.
"""

from argparse import ArgumentParser
import os
import sys
import shutil
import itertools
import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial
import pandas as pd
try:
    import pyhull
    PYHULL_INSTALLED = True
except ImportError:
    PYHULL_INSTALLED = False

import distributions

parser = ArgumentParser()
parser.add_argument("--sources-file",
                    help="The file which contains the data",
                    dest="sources_file",
                    type=str,
                    default=None)
parser.add_argument("--method",
                    help="The method for producing quantile regions."
                         "Can be one of the following methods:"
                         "mahal: uses mahalanobis distance to produce regions"
                         "halfspace: Uses halfspace peeling to produce regions"
                         "direct: Directly peels off convex hulls to produce regions",
                    dest="method",
                    type=str)
parser.add_argument("--write-file",
                    help="The name of the file to write the convex sets which define the regions",
                    dest="write_file",
                    type=str,
                    default='prediction_regions.csv')
parser.add_argument("--write-directory",
                    help="The name of the directory which will contain the written files and plots",
                    dest="write_directory",
                    type=str)
parser.add_argument("--alpha",
                    help="The alpha level for the prediction region"
                         "If it is unspecified, it will produce every hull"
                         "for alpha=.01-.99.",
                    dest="alpha",
                    type=float,
                    default=None)

parser.add_argument("--use-pyhull",
                    help="If this option is set, then the program will use the implementation"
                         "of the halfspace algorithm which is dependent on pyhull. If it is not"
                         "set, it will try to use the scipy implementation",
                    dest="use_pyhull",
                    action="store_true")

def main():
    args = parser.parse_args()
    if os.path.isdir(args.write_directory):
        print("To run this program, this script will delete directory {}".format(args.write_directory))
        response = input("Is this okay (y/n)? ")
        while response not in 'yn':
            response = input("Please answer with one of y or n: ")
        if response == 'y':
            shutil.rmtree(args.write_directory)
        else:
            sys.exit()
    os.mkdir(args.write_directory)

    if args.use_pyhull and not PYHULL_INSTALLED:
        print("Cannot find the module pyhull, please make sure it is installed in the proper location.")
        print("Will attempt to use the scipy implementation")
        args.use_pyhull = False

    if not args.use_pyhull:
        global PYHULL_INSTALLED
        PYHULL_INSTALLED = False
        scipy_version = int(scipy.__version__.split('.')[1])
        if scipy_version < 19:
            print("Cannot use scipy implementation as your version of scipy is too old")
            print("To use the scipy implementation, you must update to at least version 0.19.0")
            sys.exit(1)

    with open(args.sources_file) as f:
        dataframe = pd.read_csv(f, header=None)
    values = dataframe.values

    print("Constructing Regions")
    write_and_plot_regions(values, args.method, args.write_directory, args.write_file, args.alpha)


def write_and_plot_regions(points, method, write_directory, write_file, alpha=None):
    """
    Delegates the work of generating regions to one of three methods

    Args:
        points (np.ndarray): A numpy array of points n x p where n is number of points, p is dimension of points
        method (str): One of 'mahal', 'halfspace', 'direct'
        write_directory (str): The directory which will contain the write files
        write_file (str): The name of the file which contains the region information
        alpha (float): The desired alpha level for the prediction region
    """
    if method == 'mahal':
        mahalanobis_regions(points, write_directory, write_file, alpha)
    elif method == 'halfspace':
        halfspace_regions(points, write_directory, write_file, alpha)
    elif method == 'direct':
        direct_regions(points, write_directory, write_file, alpha)
    else:
        print('The method for producing regions is either unfilled or unrecognized'
              '--method must be set to one of mahal, halfspace, or direct'
              'to produce regions')
        sys.exit(1)


def mahalanobis_regions(points, write_directory, write_file, alpha=None):
    """
    Computes the respective mahalanobis region for the specified alpha.
    Then it writes the the representative points of the convex hull to the specified file

    Args:
        points (np.ndarray): A numpy array of points n x p where n is number of points, p is dimension of points
        write_directory: The directory which will contain the write files
        write_file: The name of the file which contains the region information
        alpha: The desired alpha level for the prediction region, if None, then calculates for alpha=1,..,100
    """
    region = distributions.MahalanobisRegion(points)
    peels = []

    if alpha is not None:
        region.set_region(alpha)
        hull = region.hull
        list_of_points = region.curr_points
        realized_alpha = region.realized_alpha
        peels = [hull.points[hull.vertices]]
        alphas = [realized_alpha]
    else:
        alphas = []
        for i in range(1, 100):
            alpha = i / 100
            try:
                region.set_region(alpha)
            except ValueError: # cannot make any more regions
                break
            hull = region.hull
            realized_alpha = region.realized_alpha
            hull_points = hull.points[hull.vertices]
            peels.append(hull_points)
            alphas.append(realized_alpha)
        list_of_points = region.all_points

    plot_regions(peels, list_of_points, write_directory, 'mahalanobis_region')
    write_regions(peels, alphas, write_directory, write_file)


def halfspace_regions(points, write_directory, write_file, alpha=None):
    """
    Computes the respective halfspace peel for the specified alpha.
    Then it writes the the representative points of the convex hull to the specified file

    Args:
        points (np.ndarray): A numpy array of points n x p where n is number of points, p is dimension of points
        write_directory (str): The directory which will contain the write files
        write_file (str): The name of the file which contains the region information
        alpha (float): The desired alpha level for the prediction region, if None, then computes all convex hulls
    """

    region = distributions.HalfspaceDepthRegion(points)

    if PYHULL_INSTALLED:
        distr = distributions.MultivariateEmpriicalDistribution(points, raw_data=True)
        if alpha is not None:
            list_of_points, hull, _, realized_alpha = distr.halfspacedepth_quantile_region(alpha)
            peels = [hull.points[hull.vertices]]
            alphas = [realized_alpha]
        else:
            distr.halfspacedepth_quantile_region(1)
            peels = distr.allhulls
            alphas = distr.alphas
            list_of_points = np.array(distr.data_matrix)
    else:
        if alpha is not None:
            region.set_region(alpha)
            peels = [region.hull.points[region.hull.vertices]]
            alphas = [region.realized_alpha]
            list_of_points = region.all_points
        else:
            region.set_region(1)
            peels = region.all_hulls
            alphas = region.all_alphas
            list_of_points = region.all_points

    plot_regions(peels, list_of_points, write_directory, 'halfspace_region')
    write_regions(peels, alphas, write_directory, write_file)


def direct_regions(points, write_directory, write_file, alpha=None):
    """
    Computes the respective direct convex hull peel for the specified alpha.
    Then it writes the the representative points of the convex hull to the specified file

    Args:
        points (np.ndarray): The fitted multivariate empirical distribution to the data
        write_directory: The directory which will contain the write files
        write_file: The name of the file which contains the region information
        alpha: The desired alpha level for the prediction region, if None, then computes all convex hulls
    """
    region = distributions.DirectRegion(points)

    if PYHULL_INSTALLED:
        distr = distributions.MultivariateEmpiricalDistribution(points, raw_data=True)
        if alpha is not None:
            list_of_points, hull, _, realized_alpha = distr.direct_convex_hull_quantile_region(alpha)
            peels = [hull.points[hull.vertices]]
            alphas = [realized_alpha]
        else:
            distr.direct_convex_hull_quantile_region(1)
            peels = distr.allhulls
            alphas = distr.alphas
            list_of_points = np.array(distr.data_matrix)
    else:
        if alpha is not None:
            region.set_region(alpha)
            peels = [region.hull.points[region.hull.vertices]]
            alphas = [region.realized_alpha]
            list_of_points = region.all_points
        else:
            region.set_region(1)
            peels = region.all_hulls
            alphas = region.all_alphas
            list_of_points = region.all_points

    plot_regions(peels, list_of_points, write_directory, 'direct_region')
    write_regions(peels, alphas, write_directory, write_file)


def plot_regions(peels, points_in_hull, plot_directory, plot_file):
    """
    Plots each of the hulls in argument peels to the directory plot_directory
    with name plot_file

    Args:
         peels: A list of peels (each is a list of points in defining polytope)
         points_in_hull: list of the points
         plot_directory: directory name to store the plots
         plot_file: the name of the plot file
    """
    print("Plotting to {}".format(plot_directory))

    if not os.path.isdir(plot_directory):
        os.mkdir(plot_directory)

    for i, peel in enumerate(peels):

        hull = scipy.spatial.ConvexHull(peel)
        vertices = hull.points[hull.vertices]

        for j, comb in enumerate(itertools.combinations(range(len(vertices[0])), 2)):
            dimensions = list(zip(*vertices))
            xs, ys = dimensions[comb[0]], dimensions[comb[1]]
            projection_hull = scipy.spatial.ConvexHull(list(zip(xs, ys)))
            projection_hull_vertices = projection_hull.points[projection_hull.vertices]
            xs, ys = zip(*projection_hull_vertices)
            plt.figure(j)
            plt.plot(xs + (xs[0],), ys + (ys[0],), 'k-')
            real_points = list(zip(*points_in_hull))
            xs, ys = real_points[comb[0]], real_points[comb[1]]
            plt.plot(xs, ys, 'b.')

    for i, comb in enumerate(itertools.combinations(range(len(vertices[0])), 2)):
        plt.figure(i)
        dim1, dim2 = comb[0] + 1, comb[1] + 1  # change to 1 indexing
        plt.title("{} Projection onto dimension {} versus dimension {}".format(plot_file, dim1, dim2))
        plt.xlabel("Dimension " + str(dim1))
        plt.ylabel("Dimension " + str(dim2))
        plt.savefig(plot_directory + os.sep + plot_file + str(dim1) + 'vs' + str(dim2) + '.png')


def write_regions(peels, alphas, write_directory, write_file):
    """
    This function writes out the prediction regions in two formats.
    One is a json file and the other is a structured text file.
    Each of the regions contain an alpha, the points, and the volume as fields.

    Args:
        peels: A list of peels (each is a list of points in defining polygon)
        alphas: A list of alphas corresponding to the peels
        write_directory: The directory which will contain the write files
        write_file: The name of the json file
    """
    print("Writing to {}".format(write_directory + os.sep + write_file))
    list_of_regions = []
    for i, peel in enumerate(peels):
        hull = scipy.spatial.ConvexHull(peel)
        vertices = hull.points[hull.vertices].tolist()
        region = {}
        region['points'] = vertices
        region['alpha'] = alphas[i]
        region['volume'] = hull.volume
        list_of_regions.append(region)

    with open(write_directory + os.sep + write_file, 'w') as f:
        for region in list_of_regions:
            f.write('Alpha: {}\n'.format(region['alpha']))
            f.write('Volume: {}\n'.format(region['volume']))
            f.write('Points:\n')
            for point in region['points']:
                f.write(','.join(map(str, point)) + '\n')
            f.write('\n')


if __name__ == '__main__':
    main()
