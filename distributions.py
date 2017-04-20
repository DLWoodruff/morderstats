"""
distributions.py
  This module exports an abstract base class Distribution, which models the attributes of a probability distribution
  and lists methods (cdf, pdf, inverse cdfs) which should be present for any distribution.
  It also defines a collection of basic distributions (empirical, normal, student, etc.) which can either be
  fitted from data or parametrized.
"""

import os
from operator import itemgetter
import itertools
import copy

import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt


try:
    import pyhull.halfspace as phhs
    PYHULL_INSTALLED = True
except ImportError:
    PYHULL_INSTALLED = False


class Distribution:
    """
    This is the abstract base class for probability distributions.
    This ABC should never be instantiated and should only be used for inheritance.
    """

    def pdf(self, point):
        """
        This function accepts a single point and returns the probability
        distribution function evaluated at that point.
        This function could be modified to accept multiple points and return multiple function values
        """
        pass

    def cdf(self, point):
        """
        This function accepts a single point and returns the cumulative
        distribution function evaluated at that point.
        This function could be modified to accept multiple points and return multiple function values
        """
        pass

    def inverse_cdf(self, point):
        """
        This function accepts a point and returns the inverse cumulative distribution
        function evaluated at this point.
        It may be better to modify this function so that it evaluates at multiple points
        """
        pass


class MultivariateEmpiricalDistribution(Distribution):
    """
    This class will fit an empirical distribution to a set of given vectors.
    Multidimensional prediction intervals will be understood as prediction regions that are either
    defined by the mahalanobis distance from the mean value or calculated by a convex hull peeling algorithm.
    """
    def __init__(self, error_data, raw_data=False):
        """
        This constructor has a dictionary of dictionary as its input. It will create a
        numpy matrix, that features all the segmented data with each column representing one dimension.

        Args:
            error_data: dictionary of error dictionaries.
                         The outer indexes are the name of the dimension obtained from
                         the name of the source of uncertainty.
                         Alternatively, can be a numpy array cf points (must pass in with raw_data=True)
            raw_data: True if using a numpy array, false if using a dict of dicts
        """
        if raw_data:
            self.data_matrix = np.matrix(error_data)
            self._n, self._p = error_data.shape

            self.allhulls = []
            self.alphas = []
        else:
            self._p = len(error_data)
            self._names = list(error_data.keys())
            list_of_lists = []  # prep for numpy
            for dim in error_data:
                list_of_lists.append(list(error_data[dim].values()))
            self.data_matrix = np.matrix(list_of_lists).T  # dimensions as columns
            self._n = int(self.data_matrix.size / self._p)
            # allhulls will be overwritten by the direct and halfspace quantile region functions respectively.
            self.allhulls = []
            self.alphas = []

    def compute_mean(self):
        """
        This function will compute the mean of the distribution.
        """
        mean = np.mean(self.data_matrix, axis=0)
        return mean

    def compute_mean_from_hyperplanes(self, hyperplanes):
        """
        This function will compute the mean of each hull.
        """
        # Let's start with an empty list.
        points = []
        # Now we look at each hyperplane.
        for plane in hyperplanes:
            # The hyperplanes have a format that consists of points and in the end two other values that we don't need.
            # Now it is only points in each plane we can look at.
            for point in plane[:-2]: # last 2 values in plane are count on either side and normal vector
                # Our list of points now appends each point.
                points.append(point)
        hyperplanematrix = np.array(points)
        mean = np.mean(hyperplanematrix, axis=0)
        return mean

    def compute_covariance(self):
        """
        This function will compute the covariance of the distribution.
        """
        covariance = np.cov(self.data_matrix.T)
        return covariance

    def direct_convex_hull_quantile_region(self, alpha):
        """
        This function returns a list of at least _p+1 _p-dimensional points that define the region.
        This function calculates the convex hull, removes it and checks whether or not the new region has less points,
        specified with the set alpha. If not it calculates the convex hull of the surviving points and continues
        until it reaches a number of points less than (1 - alpha) * number of points the algorithm started with.

        Args:
            alpha: The quantile that should be approximated is defined as 1 - alpha.
        """
        list_of_points = []
        j = 0
        # iterate over number of points
        while j != self._n:
            list_of_points.append([])
            # iterate over dimensions
            for dim in range(self._p):
                list_of_points[j].append(self.data_matrix[j, dim])
            j += 1
        if list_of_points == []:
            return None, None, None, None
        else:
            # alpha = 0 displays the convex hull of the complete data.
            if alpha == 0:
                np_points = np.array(list_of_points)
                try:
                    hull = scipy.spatial.ConvexHull(list_of_points)
                except scipy.spatial.qhull.QhullError:
                    print("Skipping time, because your data can not produce a convex hull. "
                          "You provided %d points for %d dimensions. "
                          "If the first number is much larger than the second, you might have corrupted data. "
                          "If this occurs on every datetime, you might be comparing data with itself."
                          % (len(list_of_points), self._p))
                return np_points, hull, list_of_points, 1
            i = 0
            step = 1
            points = list_of_points.copy()
            np_points = np.array(points)
            self.allhulls.append(np_points)
            self.alphas.append(1)
            while i < step:

                oldnppoints = np_points
                try:
                    hull = scipy.spatial.ConvexHull(points)
                except scipy.spatial.qhull.QhullError:
                    print("Skipping time, because your data can not produce a convex hull. "
                          "You provided %d points for %d dimensions. "
                          "If the first number is much larger than the second, you might have corrupted data. "
                          "If this occurs on every datetime, you might be comparing data with itself."
                          % (len(points), self._p))
                    return None, None, None, None
                except IndexError:
                    print("Skipping time, because your data can not produce a convex hull. "
                          "You provided %d points for %d dimensions. "
                          % (len(points), self._p))
                    return None, None, None, None
                oldhull = hull
                oldpoints = points.copy()

                hull = scipy.spatial.ConvexHull(points)

                hullpoints = []
                # hullpoints will have all of the points only on the edge of the outer convex hull.
                for point in hull.vertices:
                    hullpoints.append(points[point])
                j = 0
                for point in points:
                    if point in hullpoints:
                        points[j] = 'obsolete'
                    j += 1
                while 'obsolete' in points:
                    points.remove('obsolete')
                # check if we need to peel more or are finished.
                self.alphas.append(len(points) / len(list_of_points))
                if len(points) > (1 - alpha) * len(list_of_points):
                    step += 1

                # if the old hull was closer to the quantile than the new hull, we should use the old hull.
                if abs(1 - len(oldpoints)/len(list_of_points) - alpha) < abs(1 - len(points)/len(list_of_points) - alpha):
                    realized_alpha = len(oldpoints)/len(list_of_points)
                    print("realized alpha:", round(1 - realized_alpha, 2))
                    return oldnppoints, oldhull, oldpoints, 1 - realized_alpha

                try:
                    hull = scipy.spatial.ConvexHull(points)
                except scipy.spatial.qhull.QhullError:
                    print("Skipping time, because your data can not produce a convex hull. "
                          "You provided %d points for %d dimensions. "
                          "If the first number is much larger than the second, you might have corrupted data. "
                          "If this occurs on every datetime, you might be comparing data with itself."
                          % (len(points), self._p))
                    return None, None, None, None
                except IndexError:
                    print("Skipping time, because your data can not produce a convex hull. "
                          "You provided %d points for %d dimensions. "
                          % (len(points), self._p))
                    return None, None, None, None
                np_points = np.array(points)
                self.allhulls.append(np_points)
                i += 1
            realized_alpha = len(points) / len(list_of_points)
            print("realized alpha:", round(1 - realized_alpha, 2))
            return np_points, hull, points, 1 - realized_alpha

    def mahalanobis_quantile_region(self, alpha):
        """
        This function returns a list of at least _p+1 _p-dimensional points that define the region.
        This function uses the mahalanobis distance to find the furthest points from the mean.
        The mahalanobis distance is a value that will be temporarily remembered for every point.
        These points can then be removed from the list of points.
        The convex hull of the remaining points will represent an approximation of the desired quantile region.

        Args:
            alpha: The quantile that should be approximated is defined as 1 - alpha.
        """
        mean = self.compute_mean()
        cov_mat = self.compute_covariance()
        list_of_points = []
        j = 0
        # iterate over number of points
        while j != self._n:
            list_of_points.append([])
            # iterate over dimensions
            for dim in range(self._p):
                list_of_points[j].append(self.data_matrix.item((j, dim)))
            point = list_of_points[j]
            list_of_points[j].append(scipy.spatial.distance.mahalanobis(mean, point, cov_mat))
            j += 1
        all_points = len(list_of_points)
        if list_of_points == []:
            return None, None, None, None
        else:
            # sort by mahalanobis distance
            list_of_points = sorted(list_of_points, key=itemgetter(len(list_of_points[0])-1))
            # delete the mahalanobis distance information from the points again
            for i in list_of_points:
                del i[-1]
            # del list_of_points[-0:] seems to be not consistent with doing nothing. It instead deletes everything.
            # So we have a special case here.
            if round(alpha * float(len(list_of_points))) != 0:
                del list_of_points[- round(alpha * float(len(list_of_points))):]
            # numpy points can be plotted
            if len(list_of_points) < self._p + 1:
                print("Skipping time, because you do not have enough available points to produce a convex hull. "
                      "You need at least %d points, but you only supplied %d points"
                      % ((self._p + 1), len(list_of_points)))
                return None, None, None, None
                # raise RuntimeError("""
                # You do not have enough available points to produce a convex hull.
                # You need at least %s points, but you only supplied %s points"""
                #                    % (str(self._p + 1), str(len(list_of_points))))
            else:
                np_points = np.array(list_of_points)
                try:
                    hull = scipy.spatial.ConvexHull(np_points)
                except scipy.spatial.qhull.QhullError:
                    print("Skipping time, because your data can not produce a convex hull. "
                          "You provided %d points for %d dimensions. "
                          "If the first number is much larger than the second, you might have corrupted data. "
                          "If this occurs on every datetime, you might be comparing data with itself."
                          % (len(list_of_points), self._p))
                    return None, None, None, None
                realized_alpha = len(list_of_points)/all_points
                print("realized alpha:", round(1 - realized_alpha, 2))
                return np_points, hull, list_of_points, 1 - realized_alpha

    def get_hyperplanes(self, pointslist):
        """
        this auxiliary function takes every point and creates a list of all the hyperplanes
        that can be created with these points.

        Args:
            pointslist: list of all points to be used to calculate hyperplanes.
        """
        # list_of_hyperplanes will be all combination of points. A hyperplane is therefor defined by its _n points.
        # We need this to later get all intersections of hyperplanes for the indexing.
        hyperplanes = list(itertools.combinations(pointslist, self._p))
        i = 0
        # getting rid of arrays
        while i < len(hyperplanes):
            hyperplanes[i] = list(hyperplanes[i])
            i += 1
        i = 0

        # The list_of_hyperplanes will now get more information,
        # 1. an index as used in the algorithm by Eddy (lower number of points on the side of the hyperplane)
        # 2. the normal, with which we can later calculate the intersections.
        while i < len(hyperplanes):
            try:
                side_1, side_2, normal_vector = (self.count_on_either_side(hyperplanes[i], pointslist))
            except np.linalg.linalg.LinAlgError:
                i += 1
                continue
            hyperplanes[i].append(min(side_1, side_2))
            hyperplanes[i].append(normal_vector)
            i += 1
        return hyperplanes

    def get_halfspacehull(self, halfspaces):
        """
        This auxiliary function uses the pyhull module to calculate the intersections of hyperspaces.
        It currently uses the mean, to check what side of the hyperplane to extend into the halfspace.
        Args:
            halfspaces: list of halfspaces defined by _p points, their index and a normal vector.
        """
        # this will probably not work.
        phhs_halfspaces = []
        point_in_center = np.array(self.compute_mean_from_hyperplanes(halfspaces).tolist())
        for halfspace in halfspaces:
            basis = []
            del halfspace[-2]
            iter_basis = 0
            while iter_basis < self._p - 1:
                basis.append([])
                basis_vector = []
                if halfspace[-1][0] != 0:
                    basis_vector.append((- halfspace[-1][iter_basis + 1]) / halfspace[-1][0])
                    rest_of_vector = self._p - 1
                    while rest_of_vector > 0:
                        if rest_of_vector + iter_basis == self._p - 1:
                            basis_vector.append(1)
                        else:
                            basis_vector.append(0)
                        rest_of_vector -= 1

                else:
                    basis_vector.append(1)
                    basis_vector.append((- halfspace[-1][iter_basis]) / halfspace[-1][1])
                    rest_of_vector = self._p - 2
                    while rest_of_vector > 0:
                        if rest_of_vector + iter_basis == self._p - 1:
                            basis_vector.append(1)
                        else:
                            basis_vector.append(0)
                        rest_of_vector -= 1
                basis[-1] = basis_vector
                iter_basis += 1
            phhs_halfspace = phhs.Halfspace.from_hyperplane(np.array(basis),np.array(halfspace[0]),point_in_center)
            phhs_halfspaces.append(phhs_halfspace)
        Intersection = phhs.HalfspaceIntersection(phhs_halfspaces, point_in_center)
        return Intersection.vertices.tolist()

    def halfspacedepth_quantile_region(self, alpha):
        """
        This function returns a list of at least _p+1 _p-dimensional points (may involve virtual points)
        that define the region. The region is a hull object that uses virtual points and data points.
        These virtual points are not part of the original set of points.
        This function uses a convex hull peeling algorithm as described by Eddy.
        The minimum of the number of points on either side of the hyperplane is a value that will be temporarily
        remembered for every hyperplane. For a more detailed documentation of how this algorithm works,
        please refer to the documentation in doc/convex_hull_peeling

        Args:
            alpha: The quantile that should be approximated is defined as 1 - alpha.
        """
        list_of_points = []
        j = 0
        # iterate over number of points
        while j != self._n:
            list_of_points.append([])
            # iterate over dimensions
            for dim in range(self._p):
                # We round the data here to prevent numerical errors to appear further down the code.
                list_of_points[j].append(round(self.data_matrix.item((j, dim)), 4))
            j += 1
        # we now have list_of_points, which has all of the points available
        list_of_hyperplanes = self.get_hyperplanes(list_of_points)
        # We are now prepared for the iteration. We start with an iterator for the hull. That means we create a hull
        # with the convex hull peeling algorithm, then check if the hull is what we wanted.
        # If not, we produce a smaller hull by increasing iter_hull. If, yes, we terminate.
        iter_hull = 1
        steps = 1
        # alpha = 0 gives all points. Everything else needs at least one iteration.
        realised_quantile = 1
        if alpha != 0:
            steps += 1
        nppoints = np.array(list_of_points)
        try:
            hull = scipy.spatial.ConvexHull(list_of_points)
        except scipy.spatial.qhull.QhullError:
            print("Skipping time, because your data can not produce a convex hull. "
                  "You provided %d points for %d dimensions. "
                  "If the first number is much larger than the second, you might have corrupted data. "
                  "If this occurs on every datetime, you might be comparing data with itself."
                  % (len(list_of_points), self._p))
            return None, None, None, None
        self.allhulls.append(nppoints)
        self.alphas.append(1)
        points_to_display = list_of_points
        quantilenow = 1
        # Algorithm starts here.
        while iter_hull < steps:
            oldhull = hull
            oldnppoints = nppoints
            oldpoints_to_display = points_to_display
            # This hull decreases in area by every step, because we remove points
            # and only add points that are within the previous hull.
            try:
                scipy.spatial.ConvexHull(points_to_display)
            except scipy.spatial.qhull.QhullError:
                print("Skipping time, because your data can not produce a convex hull. "
                      "You provided %d points for %d dimensions. "
                      "If the first number is much larger than the second, you might have corrupted data. "
                      "If this occurs on every datetime, you might be comparing data with itself."
                      % (len(points_to_display), self._p))
                return None, None, None, None

            relevant_hyperplanes = []
            for hyperplane in list_of_hyperplanes:
                if hyperplane[-2] == iter_hull:
                    relevant_hyperplanes.append(hyperplane)

            points = self.get_halfspacehull(relevant_hyperplanes)

            try:
                hull = scipy.spatial.ConvexHull(points)
            except scipy.spatial.qhull.QhullError:
                print("Skipping time, because your data can not produce a convex hull. "
                      "You provided %d points for %d dimensions. "
                      "If the first number is much larger than the second, you might have corrupted data. "
                      "If this occurs on every datetime, you might be comparing data with itself."
                      % (len(points_to_display), self._p))
                return None, None, None, None

            # this following block of code checks if a point should be displayed in the end for every point.
            # this code does not have to be done for alpha == None, but is so quick that it will go thru it anyway.
            points_to_display = []
            for point in list_of_points:
                point_checker = copy.deepcopy(points)
                point_checker.append(point)
                try:
                    point_checker_hull = scipy.spatial.ConvexHull(point_checker)
                except scipy.spatial.qhull.QhullError:
                    print("Skipping time, because your data can not produce a convex hull. "
                          "You provided %d points for %d dimensions. "
                          "If the first number is much larger than the second, you might have corrupted data. "
                          "If this occurs on every datetime, you might be comparing data with itself."
                          % (len(points_to_display), self._p))
                    return None, None, None, None
                if point_checker_hull.area == hull.area:
                    points_to_display.append(point)

            # for alpha == None:
            self.allhulls.append(np.array(points))
            self.alphas.append(len(points_to_display) / len(list_of_points))
            # check if we are finished
            quantilenow = float(len(points_to_display)) / float(len(list_of_points))
            if quantilenow > 1 - alpha:
                steps += 1
            iter_hull += 1
            nppoints = np.array(points)

            # if the old hull was closer to the quantile than the new hull, we should use the old hull.
            if abs(1 - realised_quantile - alpha) < abs(1 - quantilenow - alpha) and quantilenow <= 1 - alpha:
                print("realized alpha:", round(1 - realised_quantile, 2))
                return oldnppoints, oldhull, oldpoints_to_display, 1 - realised_quantile
            realised_quantile = float(len(points_to_display)) / float(len(list_of_points))
        print("realized alpha:", round(1 - quantilenow, 2))
        return nppoints, hull, points_to_display, 1 - quantilenow

    def count_on_either_side(self, points_on_hyperplane, all_points):
        """
        This function calculates the number of points on either side of a hyperplane.
        It also checks if the hyperplane was degenerate (and returns (-1, -1) if it was).

        points_on_hyperplane: This is a list of lists. Each list represents a point that consists
                              of the coordinates in the different dimensions.
        all_points: This list of lists consists of all points (even the ones that make up the hyperplane) to check.
        """
        points = copy.deepcopy(points_on_hyperplane)
        normal_vector = []
        if len(points) > self._p:
            normal_vector = np.array(points[-1])
            points.pop()
            points.pop()
        hyperplane_mat = np.array(points)
        if normal_vector == []:
            one_vector = [1] * self._p
            one_vector = np.array(one_vector)
            # if this gives an error, we have to check the side.
            # Add this functionality, the first time it crashes on the next line.
            normal_vector = np.linalg.solve(hyperplane_mat, one_vector)
        side_a = 0
        side_b = 0
        for point in all_points:
            first_axis = 0

            j = 1
            while j < self._p:
                first_axis -= normal_vector[j] * (point[j] - hyperplane_mat[0][j])
                j += 1
            first_axis /= normal_vector[0]
            first_axis += hyperplane_mat[0][0]
            if float(first_axis) - 0.00005 < point[0] < float(first_axis) + 0.00005:
                continue
            elif float(first_axis) > point[0]:
                side_a += 1
            elif float(first_axis) < point[0]:
                side_b += 1
        return side_a, side_b, normal_vector.tolist()

    def check_convex_hull(self):
        a_1, b_1, _, _ = self.halfspacedepth_quantile_region(0)
        a_2, b_2, _, _ = self.mahalanobis_quantile_region(0)
        b_1_list = []
        b_2_list = []
        for index in b_1.vertices:
            b_1_list.append(b_1.points[index].tolist())
        for index in b_2.vertices:
            b_2_list.append(b_2.points[index].tolist())
        if (sorted(a_1.tolist())) == (sorted(a_2.tolist())) and sorted(b_1_list) == sorted(b_2_list):
            pass
        else:
            print("Warning: The convex hull of the two possible methods does not yield the same result.")


class Halfspace:
    """
    This class is a representation of a halfspace built using scipy libraries
    Internally, it stores n points to determine an n-dimensional hyperplane
    as well as an interior point to determine which half of n-space
    it is referring to.

    Attributes:
        normal_vector: a n-vector which is orthogonal to the hyperplane
        point: An arbitrary point on the surface of the halfspace
        n: the number of dimensions
        interior_point: an n-vector referring to a point in the interior of the halfspace

    Args:
        point_array: An n x n array containing n points to construct the hyperplane from
        interior_point: A n-vector referring to a point in the halfspace
    """
    def __init__(self, point_array, interior_point):
        self.n = self.ndim = len(point_array[0])
        if self.n != len(point_array):
            print("To form a hyperplane in {} dimensions, you must pass in {} points".format(self.n, self.n))
            print("You passed in {} points of dimension {}".format(len(point_array), self.n))
            raise ValueError
        self.normal_vector = self._get_normal_vector(point_array)
        self.point = point_array[0]
        self.interior_point = interior_point
        self.constant = -np.dot(self.normal_vector, self.point)  # d in ax + by + cz + d <= 0
        if np.dot(self.normal_vector, interior_point) + self.constant >= 0:  # This is so we have ax + by + cz + d <= 0, not >=
            self.normal_vector = -self.normal_vector
            self.constant = -self.constant

    def side_of_point(self, point, tol=1e-5):
        """
        Determines if a point is 'above', 'on', or 'below' the plane.
        'above' and 'below' is determined by the orientation of the normal vector.
        Args:
            point: A n-vector
            tol: tolerance for how close to 0 should be considered equal to 0
        Returns:
            1 if above
            0 if on
            -1 if below
        """
        vector_to_point = point - self.point
        dot_product = np.dot(self.normal_vector, vector_to_point)
        if dot_product > tol:
            return 1
        elif dot_product < -tol:
            return -1
        else:
            return 0

    def _get_normal_vector(self, point_array):
        temp_matrix = point_array[:-1] - point_array[-1]
        rank, nullspace = self._null(temp_matrix)
        if len(temp_matrix) - rank > 1:
            raise ValueError("Points passed do not form an (n-1) dimensional space and are degenerate")
        return nullspace[0]

    def _null(self, a, rtol=1e-5):
        u, s, v = np.linalg.svd(a)
        rank = (s > rtol*s[0]).sum()
        return rank, v[rank:].copy()

    def __str__(self):
        rhs = np.dot(self.normal_vector, self.point)
        string = "Hyperplane: "
        for i, num in enumerate(self.normal_vector[:-1]):
            string += "{:.4f}x{} + ".format(num, i+1)
        string += "{:.4f}x{}".format(self.normal_vector[-1], self.n)
        string += " <= {:.4f}".format(rhs)
        return string

    def in_plane(self, point):
        """
        Returns true if the point is in the hyperplane

        Args:
            point (np.ndarray): A numpy array with n components

        Returns:
            bool: True if point is in plane, False otherwise
        """

        return np.isclose(np.dot(point - self.point, self.normal_vector), 0)

    def in_halfspace(self, point):
        """
        Returns true if the point is in the halfspace

        Args:
            point (np.ndarray): A numpy array with n components

        Returns:
            bool: True if point is in halfspace, False otherwise
        """

        return np.dot(self.normal_vector, point) <= self.constant

    def as_array(self):
        """
        If the halfspace is defined by equation a1x1 + a2x2 + ... + anxn + b <= 0
        Writes the halfspace in notation [a1 a2 a3 ... an b]

        Returns:
            np.ndarray: A (n+1)-vector representation of the halfspace
        """
        return np.append(self.normal_vector, self.constant)


class Region:
    """
    This class should be a generic class for fitting convex regions
    to a collection of points.
    """
    def __init__(self, points):
        self.all_points = np.array(points)
        if self.all_points.ndim != 2:
            raise ValueError("The points array passed in should be a list of points or a 2-D array")
        self.ndim = self.all_points.shape[1]
        self.num_points = len(points)
        self.hull = scipy.spatial.ConvexHull(self.all_points)
        self._vertex_indices = self.hull.vertices
        self.points_in_hull = self.all_points
        self.realized_alpha = 0

    def plot(self, name, directory=None, title=None):
        """
        Plots the current region to a file with specified name and directory
        Saves to a png file
        Args:
            name (str): name of the file containing plot excluding extension
            directory (str): name of directory to save the file to
            title (str): title of plot
        """

        if directory is None:
            directory = os.getcwd()

        if not(os.path.isdir(directory)):
            print("Directory {} does not exist, making directory".format(directory))
            os.mkdir(directory)

        if self.ndim == 3:
            self.plot3d(name + '.png', directory)

        vertices = self.hull.points[self.hull.vertices]

        for i, comb in enumerate(itertools.combinations(range(self.ndim), 2)):
            plt.figure()
            dimensions = vertices.T
            xs, ys = dimensions[comb[0]], dimensions[comb[1]]
            projection_hull = scipy.spatial.ConvexHull(list(zip(xs, ys)))
            projection_hull_vertices = projection_hull.points[projection_hull.vertices]
            xs, ys = self.all_points[:,comb[0]], self.all_points[:,comb[1]]
            plt.plot(xs, ys, 'b.')
            xs, ys = zip(*projection_hull_vertices)
            plt.plot(xs + (xs[0],), ys + (ys[0],), 'k-')
            dim1, dim2 = comb[0] + 1, comb[1] + 1  # change to 1 indexing

            if title is None:
                plt.title("{} Projection onto dimension {} versus dimension {}".format(name, dim1, dim2))
            else:
                plt.title(title)

            plt.savefig(directory + os.sep + name + str(dim1) + 'vs' + str(dim2) + '.png')
            plt.close()

    def plot3d(self, filename, directory=None):
        """
        Plot 3 dimensional regions to a 3 dimensional plot

        Args:
            filename (str): The path to where the plot should be saved
            directory (str): The name of the directory to store the file
        """

        if directory is None:
            directory = os.getcwd()

        if self.ndim != 3:
            raise RuntimeError("The dimensionality of the points must be 3 dimensional")

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for point in self.all_points:
            if in_hull(point, self.hull):
                plt.plot([point[0]], [point[1]], [point[2]], 'r.', alpha=0.4)
            else:
                plt.plot([point[0]], [point[1]], [point[2]], 'r.')

        polygons = []

        for simplex in self.hull.simplices:
            polygon = self.hull.points[simplex]
            polygons.append(polygon)

        polygon_collection = Poly3DCollection(polygons)
        polygon_collection.set_alpha(0.5)

        ax.add_collection3d(polygon_collection, zs='z')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('Convex Region for alpha = {:.2f}'.format(self.realized_alpha))

        plt.savefig(directory + os.sep + filename)

    def equals_hull(self, other):
        """
        Returns true if convex hull vertices are the same.

        Args:
            other (Region): Another Region object

        Returns:
            bool: true if hull vertices are the same up to reordering
        """
        vertices1 = self.hull.points[self.hull.vertices]
        vertices2 = other.hull.points[other.hull.vertices]

        sorted_vertices1 = np.sort(vertices1, axis=0)
        sorted_vertices2 = np.sort(vertices2, axis=0)

        return np.allclose(sorted_vertices1, sorted_vertices2)


class RegionSequence(Region):
    """
    This class
    """
    def __init__(self, points):
        Region.__init__(self, points)
        self.curr_points = self.all_points
        self.all_hulls = [self.hull.points[self.hull.vertices]]
        self.all_alphas = [0]
        self.curr_alpha = 0

    def peel(self):
        raise NotImplemented("This method should be implemented in any subclass of RegionSequence")

    def plot_sequence(self, name, directory):
        """
        Plots all the already generated convex hulls to the file specified

        Args:
            name (str): name of file to save plot in excluding extension
            directory (str): Directory to store file in

        """
        plt.plot(self.all_points[:,0], self.all_points[:,1], '.')
        for i, peel in enumerate(self.all_hulls):

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
                real_points = list(zip(*self.all_points))
                xs, ys = real_points[comb[0]], real_points[comb[1]]
                plt.plot(xs, ys, 'b.')

        for i, comb in enumerate(itertools.combinations(range(len(vertices[0])), 2)):
            plt.figure(i)
            dim1, dim2 = comb[0] + 1, comb[1] + 1  # change to 1 indexing
            plt.title("{} Projection onto dimension {} versus dimension {}".format(name, dim1, dim2))
            plt.xlabel("Dimension " + str(dim1))
            plt.ylabel("Dimension " + str(dim2))
            plt.savefig(directory + os.sep + name + str(dim1) + 'vs' + str(dim2) + '.png')

        plt.savefig(name)
        plt.close()

    def set_region(self, alpha):
        """
        Finds the region in the region sequence with alpha closest to passed in alpha
        Sets internal attributes hull and realized_alpha to the corresponding values
        Peels if necessary

        Args:
            alpha (float): The desired alpha
        """

        assert 0 <= alpha <= 1

        if alpha < self.curr_alpha:
            closest_index = 0
            smallest_difference = abs(alpha - self.all_alphas[0])
            for i, alpha2 in enumerate(self.all_alphas):
                if abs(alpha2 - alpha) < smallest_difference:
                    closest_index = i
                    smallest_difference = abs(alpha2 - alpha)
            self.hull = scipy.spatial.ConvexHull(self.all_hulls[closest_index])
            self.realized_alpha = self.all_alphas[closest_index]
            return

        while alpha > self.curr_alpha:
            try:
                self.peel()
            except ValueError:
                break

        closest_index = -1 if abs(alpha - self.all_alphas[-1]) < abs(alpha - self.all_alphas[-2]) else -2

        self.hull = scipy.spatial.ConvexHull(self.all_hulls[closest_index])
        self.realized_alpha = self.all_alphas[closest_index]


class MahalanobisRegion(Region):
    """
    This class fits a convex hull containing 1-alpha percent of the points.
    The way it constructs the region is based on the mahalanobis distance
    """
    def __init__(self, points):
        Region.__init__(self, points)
        self.mean = np.mean(self.all_points, axis=0)
        self.covariance = np.cov(self.all_points.T)
        self.distances = np.array([scipy.spatial.distance.mahalanobis(point, self.mean, self.covariance)
                                    for point in self.all_points])

    def set_region(self, alpha):
        """
        Sets the hull attribute to be the convex hull of the (1-alpha)* 100% points
        that are closest the mean point according to the mahalanobis distance

        Args:
            alpha (float): The desired alpha
        """
        distance_percentile = np.percentile(self.distances, (1-alpha)*100)
        self.curr_points = self.all_points[self.distances <= distance_percentile]
        self.points_in_hull = self.curr_points

        try:
            self.hull = scipy.spatial.ConvexHull(self.curr_points)
        except scipy.spatial.qhull.QhullError:
            raise ValueError("Failure to reach alpha")

        self.realized_alpha = len(self.curr_points) / len(self.all_points)


class HalfspaceDepthRegion(RegionSequence):
    """
    This uses the algorithm described in 'Convex Hull Peeling' by W.F. Eddy
    to compute a sequence of convex regions
    """
    def __init__(self, points):
        RegionSequence.__init__(self, points)
        halfspaces = {}

        interior_point = np.mean(self.curr_points, axis=0)

        for i, combination in enumerate(itertools.combinations(self.all_points, self.ndim)):
            try:
                halfspace = Halfspace(combination, interior_point)
            except ValueError:  # hyperplane is degenerate
                continue
            count1, count2 = 0, 0
            for point in self.all_points:
                side_of_point = halfspace.side_of_point(point)
                if side_of_point == 1:
                    count1 += 1
                elif side_of_point == -1:
                    count2 += 1
            halfspace_index = min(count1, count2)
            if halfspace_index in halfspaces:
                halfspaces[halfspace_index].append(halfspace)
            else:
                halfspaces[halfspace_index] = [halfspace]

        self.halfspaces = halfspaces
        self.index = 1

    def peel(self):
        halfspaces = self.halfspaces.get(self.index, None)
        while halfspaces is None:
            self.index += 1
            halfspaces = self.halfspaces.get(self.index, None)
            if self.index > len(self.all_points) / 2:
                raise ValueError("End of sequence")

        halfspaces = np.array([halfspace.as_array() for halfspace in halfspaces])

        interior_point = self._compute_interior_point(halfspaces)

        halfspace_intersection = scipy.spatial.HalfspaceIntersection(halfspaces, interior_point)

        self.hull = scipy.spatial.ConvexHull(halfspace_intersection.intersections)

        self.curr_points = np.array([point for point in self.curr_points if in_hull(point, self.hull)])
        self.index += 1

        self.all_hulls.append(halfspace_intersection.intersections)
        self.curr_alpha = 1 - len(self.curr_points) / len(self.all_points)
        self.all_alphas.append(self.curr_alpha)

    def _compute_interior_point(self, halfspaces):
        """
        Solves a linear program to compute an interior point of the intersection of halfspaces
        The linear program is
            max y
        subject to
            Ax + y||A_i|| <= -b

        where A is composed of the normal vectors of each halfspace, A_i is the ith row and
        ||*|| is the euclidean norm
        Args:
            halfspaces: A 2-D array of halfspaces. Each row is of the form
                              [a1 a2 ... an b] where the halfspace is defined by a1x1 + a2x2 + .. + anxn + b <= 0
        Returns:
            An n-vector which is contained in the intersection of halfspaces
        """

        num_halfspaces = halfspaces.shape[0]

        # compute the norm of each normal vector for each halfspace
        row_norms = np.linalg.norm(halfspaces[:, :-1], axis=1)
        norm_vector = np.reshape(row_norms, (num_halfspaces, 1))

        A = np.hstack((halfspaces[:,:-1], norm_vector))
        b = -halfspaces[:,-1:]

        # A feasible solution to linear programming problem is [0 0 ... 0 -1]
        feasible_solution = np.zeros(self.ndim+1)
        feasible_solution[-1] = -1

        solution = scipy.optimize.linprog(feasible_solution, A_ub=A, b_ub=b)

        return solution.x[:-1]


class DirectRegion(RegionSequence):
    """
    This class constructs a nested sequence of convex hulls simply by constructing the next convex hull
    via peeling the current hull off and taking the convex hull of the remaining points.
    """
    def peel(self):
        self.curr_points = np.delete(self.curr_points, self._vertex_indices, axis=0)
        try:
            self.hull = scipy.spatial.ConvexHull(self.curr_points)
        except scipy.spatial.qhull.QhullError:
            raise ValueError("End of sequence")

        self._vertex_indices = self.hull.vertices
        vertices = self.curr_points[self.hull.vertices]
        alpha = 1 - (len(self.curr_points) / len(self.all_points))
        self.all_hulls.append(vertices)
        self.all_alphas.append(alpha)
        self.curr_alpha = alpha
        self.points_in_hull = self.curr_points


def in_hull(p, hull, tol=1e-5):
    """
    Tests if point p (a numpy array) is inside a scipy.spatial.ConvexHull object
    This uses the dot product of the normal vector of each facet of the convex hull
    and the point to determine if the point is in the correct halfspace for each hyperplane.

    Each facet has an equation of the form a1x1 + a2x2 + ... + anxn + b = 0
    If we evaluate the lefthand side of this equation and find it to be negative,
    the point is in the correct halfspace for the hyperplane.

    Args:
        p: A numpy array of shape 1 x n
        hull: A scipy.spatial.ConvexHull object
        tol: A tolerance for how to close to zero will be considered equal to zero
    """
    for equation in hull.equations:
        #  equation is a vector of the form [a1 a2 a3 ... an b]
        normal_vector = equation[:-1]
        constant_term = equation[-1]

        if np.dot(normal_vector, p) + constant_term > tol:  # on wrong side
            return False

    else:
        return True