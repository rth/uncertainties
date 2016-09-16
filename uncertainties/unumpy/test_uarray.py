# 
"""Tests suite for UncertaintyArray

Adapted from masked array unit tests (by Pierre Gerard-Marchant)
"""
from __future__ import division, absolute_import, print_function

import warnings
import pickle
import operator
import itertools
from functools import reduce


import numpy as np
import numpy.core.fromnumeric as fromnumeric
#import numpy.core.umath as umath
from numpy.testing import (
    TestCase, run_module_suite, assert_raises)
from numpy import ndarray
from numpy.compat import asbytes, asbytes_nested
from numpy.ma.testutils import (
    assert_, assert_array_equal, assert_equal, assert_almost_equal,
    assert_equal_records, fail_if_equal, assert_not_equal,
    assert_mask_equal
    )

from uncertainties.unumpy.core import uarray
from uncertainties.unumpy.core2 import (UncertainArray, uncertain,
         uncertain_array, isUncertainArray)
from unittest.case import SkipTest
#from numpy.ma.core import (
#    MAError, MaskError, MaskType, UncertainArray, abs, absolute, add, all,
#    allclose, allequal, alltrue, angle, anom, arange, arccos, arccosh, arctan2,
#    arcsin, arctan, argsort, array, asarray, choose, concatenate,
#    conjugate, cos, cosh, count, diag, divide, empty,
#    empty_like, equal, exp, flatten_mask, filled, fix_invalid,
#    flatten_structured_array, fromflex, getmask, getmaskarray, greater,
#    greater_equal, identity, inner, isUncertainArray, less, less_equal, log,
#    log10, make_mask, make_mask_descr, mask_or, uncertain, uncertain_array,
#    uncertain_equal, uncertain_greater, uncertain_greater_equal, uncertain_inside,
#    uncertain_less, uncertain_less_equal, uncertain_not_equal, uncertain_outside,
#    uncertain_print_option, uncertain_values, uncertain_where, max, maximum,
#    min, minimum, mod, multiply,
#    mvoid, nomask, not_equal, ones, outer, power, product, put, putmask,
#    ravel, repeat, reshape, resize, shape, sin, sinh, sometrue, sort, sqrt,
#    subtract, sum, take, tan, tanh, transpose, where, zeros,
#    )

pi = np.pi



class TestUncertainArray(TestCase):
    # Base test class for UncertainArrays.

    def setUp(self):
        # Base data definition.
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = np.array([1, 0, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 0])
        m2 = np.array([0, 0, 3, 0, 0, 6, 1, 0, 0, 0, 0, 1])
        xm = uncertain_array(x, m1)
        ym = uncertain_array(y, m2)
        z = np.array([-.5, 0., .5, .8])
        zm = uncertain_array(z, [0, 1, 0, 0])
        xf = np.where(m1, 1e+20, x)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)

    def test_basicattributes(self):
        # Tests some basic array attributes.
        a = uncertain_array([1, 3, 2])
        b = uncertain_array([1, 3, 2], [1, 0, 1])
        assert_equal(a.ndim, 1)
        assert_equal(b.ndim, 1)
        assert_equal(a.size, 3)
        assert_equal(b.size, 3)
        assert_equal(a.shape, (3,))
        assert_equal(b.shape, (3,))

    def test_basic0d(self):
        # Checks masking a scalar
        raise SkipTest
        x = uncertain_array(0)
        assert_equal(str(x), '0')

    def test_basic1d(self):
        # Test of basic array creation and properties in 1 dimension.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        self.assertTrue(not isUncertainArray(x))
        self.assertTrue(isUncertainArray(xm))
        s = x.shape
        assert_equal(np.shape(xm), s)
        assert_equal(xm.shape, s)
        assert_equal(xm.dtype, x.dtype)
        assert_equal(zm.dtype, z.dtype)
        assert_equal(xm.size, reduce(lambda x, y:x * y, s))
        assert_array_equal(xm, xf)
        assert_array_equal(x, xm)

    def test_basic2d(self):
        # Test of basic array creation and properties in 2 dimensions.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        for s in [(4, 3), (6, 2)]:
            x.shape = s
            y.shape = s
            xm.shape = s
            ym.shape = s
            xf.shape = s

            self.assertTrue(not isUncertainArray(x))
            self.assertTrue(isUncertainArray(xm))
            assert_equal(xm.shape, s)
            assert_equal(xm.size, reduce(lambda x, y:x * y, s))
            assert_equal(xm, xf)
            assert_equal(x, xm)

    def test_concatenate_basic(self):
        raise SkipTest
        # Tests concatenations.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # basic concatenation
        assert_equal(np.concatenate((x, y)), concatenate((xm, ym)))
        assert_equal(np.concatenate((x, y)), concatenate((x, y)))
        assert_equal(np.concatenate((x, y)), concatenate((xm, y)))
        assert_equal(np.concatenate((x, y, x)), concatenate((x, ym, x)))

    def test_concatenate_alongaxis(self):
        raise SkipTest
        # Tests concatenations.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # Concatenation along an axis
        s = (3, 4)
        x.shape = y.shape = xm.shape = ym.shape = s
        assert_equal(xm.mask, np.reshape(m1, s))
        assert_equal(ym.mask, np.reshape(m2, s))
        xmym = concatenate((xm, ym), 1)
        assert_equal(np.concatenate((x, y), 1), xmym)
        assert_equal(np.concatenate((xm.mask, ym.mask), 1), xmym._mask)

        x = zeros(2)
        y = uncertain_array(ones(2), [0, 1])
        z = concatenate((x, y))
        assert_array_equal(z, [0, 0, 1, 1])
        assert_array_equal(z.mask, [False, False, False, True])
        z = concatenate((y, x))
        assert_array_equal(z, [1, 1, 0, 0])
        assert_array_equal(z.mask, [False, True, False, False])

    def test_concatenate_flexible(self):
        raise SkipTest
        # Tests the concatenation on flexible arrays.
        data = uncertain_array(list(zip(np.random.rand(10),
                                     np.arange(10))),
                            dtype=[('a', float), ('b', int)])

        test = concatenate([data[:5], data[5:]])
        assert_equal_records(test, data)

    def test_creation_ndmin(self):
        # Check the use of ndmin
        x = uncertain_array([1, 2, 3], [1, 0, 0], ndmin=2)
        print(x._data, x._mask)
        print(x._std)
        assert False # why x has an attribute _mask?!
        assert_equal(x.shape, (1, 3))
        assert_equal(x._data, [[1, 2, 3]])
        assert_equal(x._mask, [[1, 0, 0]])

    def test_creation_ndmin_from_uncertainarray(self):
        # Make sure we're not losing the original mask w/ ndmin
        x = uncertain_array([1, 2, 3])
        x[-1] = uncertain
        xx = uncertain_array(x, ndmin=2, dtype=float)
        assert_equal(x.shape, x._mask.shape)
        assert_equal(xx.shape, xx._mask.shape)

    def test_creation_with_list_of_uncertainarrays(self):
        # Tests creating a uncertain array from a list of uncertain arrays.
        x = uncertain_array(np.arange(5), [1, 0, 0, 0, 0])
        data = uncertain_array((x, x[::-1]))
        assert_equal(data, [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        assert_equal(data._mask, [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

        x.mask = nomask
        data = uncertain_array((x, x[::-1]))
        assert_equal(data, [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        self.assertTrue(data.mask is nomask)

    def test_asarray(self):
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        xmm = asarray(xm)
        assert_equal(xmm._data, xm._data)
        assert_equal(xmm._mask, xm._mask)

    def test_asarray_default_order(self):
        # See Issue #6646
        m = np.eye(3).T
        self.assertFalse(m.flags.c_contiguous)

        new_m = asarray(m)
        self.assertTrue(new_m.flags.c_contiguous)

    def test_asarray_enforce_order(self):
        # See Issue #6646
        m = np.eye(3).T
        self.assertFalse(m.flags.c_contiguous)

        new_m = asarray(m, order='C')
        self.assertTrue(new_m.flags.c_contiguous)

    def test_fix_invalid(self):
        # Checks fix_invalid.
        with np.errstate(invalid='ignore'):
            data = uncertain_array([np.nan, 0., 1.], [0, 0, 1])
            data_fixed = fix_invalid(data)
            assert_equal(data_fixed._mask, [1., 0., 1.])

    def test_uncertainelement(self):
        # Test of uncertain element
        x = np.arange(6)
        x[1] = uncertain
        self.assertTrue(str(uncertain) == '--')
        self.assertTrue(x[1] is uncertain)
        assert_equal(filled(x[1], 0), 0)

    def test_set_element_as_object(self):
        # Tests setting elements with object
        a = empty(1, dtype=object)
        x = (1, 2, 3, 4, 5)
        a[0] = x
        assert_equal(a[0], x)
        self.assertTrue(a[0] is x)

        import datetime
        dt = datetime.datetime.now()
        a[0] = dt
        self.assertTrue(a[0] is dt)

    def test_indexing(self):
        # Tests conversions and indexing
        x1 = np.array([1, 2, 4, 3])
        x2 = uncertain_array(x1, [1, 0, 0, 0])
        x3 = uncertain_array(x1, [0, 1, 0, 1])
        x4 = uncertain_array(x1)
        # test conversion to strings
        str(x2)  # raises?
        repr(x2)  # raises?
        assert_equal(np.sort(x1), sort(x2, endwith=False))
        # tests of indexing
        assert_(type(x2[1]) is type(x1[1]))
        assert_(x1[1] == x2[1])
        assert_(x2[0] is uncertain)
        assert_equal(x1[2], x2[2])
        assert_equal(x1[2:5], x2[2:5])
        assert_equal(x1[:], x2[:])
        assert_equal(x1[1:], x3[1:])
        x1[2] = 9
        x2[2] = 9
        assert_equal(x1, x2)
        x1[1:3] = 99
        x2[1:3] = 99
        assert_equal(x1, x2)
        x2[1] = uncertain
        assert_equal(x1, x2)
        x2[1:3] = uncertain
        assert_equal(x1, x2)
        x2[:] = x1
        x2[1] = uncertain
        assert_(allequal(getmask(x2), array([0, 1, 0, 0])))
        x3[:] = uncertain_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x3), array([0, 1, 1, 0])))
        x4[:] = uncertain_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x4), array([0, 1, 1, 0])))
        assert_(allequal(x4, array([1, 2, 3, 4])))
        x1 = np.arange(5) * 1.0
        x2 = uncertain_values(x1, 3.0)
        assert_equal(x1, x2)
        assert_(allequal(array([0, 0, 0, 1, 0], MaskType), x2.mask))
        x1 = uncertain_array([1, 'hello', 2, 3], object)
        x2 = np.array([1, 'hello', 2, 3], object)
        s1 = x1[1]
        s2 = x2[1]
        assert_equal(type(s2), str)
        assert_equal(type(s1), str)
        assert_equal(s1, s2)
        assert_(x1[1:1].shape == (0,))

    def test_matrix_indexing(self):
        # Tests conversions and indexing
        x1 = np.matrix([[1, 2, 3], [4, 3, 2]])
        x2 = uncertain_array(x1, [[1, 0, 0], [0, 1, 0]])
        x3 = uncertain_array(x1, [[0, 1, 0], [1, 0, 0]])
        x4 = uncertain_array(x1)
        # test conversion to strings
        str(x2)  # raises?
        repr(x2)  # raises?
        # tests of indexing
        assert_(type(x2[1, 0]) is type(x1[1, 0]))
        assert_(x1[1, 0] == x2[1, 0])
        assert_(x2[1, 1] is uncertain)
        assert_equal(x1[0, 2], x2[0, 2])
        assert_equal(x1[0, 1:], x2[0, 1:])
        assert_equal(x1[:, 2], x2[:, 2])
        assert_equal(x1[:], x2[:])
        assert_equal(x1[1:], x3[1:])
        x1[0, 2] = 9
        x2[0, 2] = 9
        assert_equal(x1, x2)
        x1[0, 1:] = 99
        x2[0, 1:] = 99
        assert_equal(x1, x2)
        x2[0, 1] = uncertain
        assert_equal(x1, x2)
        x2[0, 1:] = uncertain
        assert_equal(x1, x2)
        x2[0, :] = x1[0, :]
        x2[0, 1] = uncertain
        assert_(allequal(getmask(x2), np.array([[0, 1, 0], [0, 1, 0]])))
        x3[1, :] = uncertain_array([1, 2, 3], [1, 1, 0])
        assert_(allequal(getmask(x3)[1], array([1, 1, 0])))
        assert_(allequal(getmask(x3[1]), array([1, 1, 0])))
        x4[1, :] = uncertain_array([1, 2, 3], [1, 1, 0])
        assert_(allequal(getmask(x4[1]), array([1, 1, 0])))
        assert_(allequal(x4[1], array([1, 2, 3])))
        x1 = np.matrix(np.arange(5) * 1.0)
        x2 = uncertain_values(x1, 3.0)
        assert_equal(x1, x2)
        assert_(allequal(array([0, 0, 0, 1, 0], MaskType), x2.mask))

    def test_copy(self):
        # Tests of some subtle points of copying and sizing.
        n = [0, 0, 1, 0, 0]
        m = make_mask(n)
        m2 = make_mask(m)
        self.assertTrue(m is m2)
        m3 = make_mask(m, copy=1)
        self.assertTrue(m is not m3)

        x1 = np.arange(5)
        y1 = uncertain_array(x1, m)
        assert_equal(y1._data.__array_interface__, x1.__array_interface__)
        self.assertTrue(allequal(x1, y1.data))
        assert_equal(y1._mask.__array_interface__, m.__array_interface__)

        y1a = uncertain_array(y1)
        self.assertTrue(y1a._data.__array_interface__ ==
                        y1._data.__array_interface__)
        self.assertTrue(y1a.mask is y1.mask)

        y2 = uncertain_array(x1, m)
        self.assertTrue(y2._data.__array_interface__ == x1.__array_interface__)
        self.assertTrue(y2._mask.__array_interface__ == m.__array_interface__)
        self.assertTrue(y2[2] is uncertain)
        y2[2] = 9
        self.assertTrue(y2[2] is not uncertain)
        self.assertTrue(y2._mask.__array_interface__ != m.__array_interface__)
        self.assertTrue(allequal(y2.mask, 0))

        y3 = uncertain_array(x1 * 1.0, m)
        self.assertTrue(filled(y3).dtype is (x1 * 1.0).dtype)

        x4 = np.arange(4)
        x4[2] = uncertain
        y4 = resize(x4, (8,))
        assert_equal(concatenate([x4, x4]), y4)
        assert_equal(getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0])
        y5 = repeat(x4, (2, 2, 2, 2), axis=0)
        assert_equal(y5, [0, 0, 1, 1, 2, 2, 3, 3])
        y6 = repeat(x4, 2, axis=0)
        assert_equal(y5, y6)
        y7 = x4.repeat((2, 2, 2, 2), axis=0)
        assert_equal(y5, y7)
        y8 = x4.repeat(2, 0)
        assert_equal(y5, y8)

        y9 = x4.copy()
        assert_equal(y9._data, x4._data)
        assert_equal(y9._mask, x4._mask)

        x = uncertain_array([1, 2, 3], [0, 1, 0])
        # Copy is False by default
        y = uncertain_array(x)
        assert_equal(y._data.ctypes.data, x._data.ctypes.data)
        assert_equal(y._mask.ctypes.data, x._mask.ctypes.data)
        y = uncertain_array(x, copy=True)
        assert_not_equal(y._data.ctypes.data, x._data.ctypes.data)
        assert_not_equal(y._mask.ctypes.data, x._mask.ctypes.data)

    def test_copy_immutable(self):
        # Tests that the copy method is immutable, GitHub issue #5247
        a = np.ma.array([1, 2, 3])
        b = np.ma.array([4, 5, 6])
        a_copy_method = a.copy
        b.copy
        assert_equal(a_copy_method(), [1, 2, 3])

    def test_deepcopy(self):
        from copy import deepcopy
        a = uncertain_array([0, 1, 2], [0, 1, 0])
        copied = deepcopy(a)
        assert_equal(copied.mask, a.mask)
        assert_not_equal(id(a._mask), id(copied._mask))

        copied[1] = 1
        assert_equal(copied.mask, [0, 0, 0])
        assert_equal(a.mask, [0, 1, 0])

        copied = deepcopy(a)
        assert_equal(copied.mask, a.mask)
        copied.mask[1] = False
        assert_equal(copied.mask, [0, 0, 0])
        assert_equal(a.mask, [0, 1, 0])

    def test_str_repr(self):
        a = uncertain_array([0, 1, 2], [0, 1, 0])
        assert_equal(str(a), '[0 -- 2]')
        assert_equal(repr(a), 'uncertain_array(data = [0 -- 2],\n'
                              '             mask = [False  True False],\n')

        a = np.arange(2000)
        a[1:50] = np.ma.uncertain
        assert_equal(
            repr(a),
            'uncertain_array(data = [0 -- -- ..., 1997 1998 1999],\n'
            '             mask = [False  True  True ..., False False False],\n'
        )

    def test_pickling(self):
        # Tests pickling
        a = np.arange(10)
        a[::3] = uncertain
        a_pickled = pickle.loads(a.dumps())
        assert_equal(a_pickled._mask, a._mask)
        assert_equal(a_pickled._data, a._data)

    def test_pickling_subbaseclass(self):
        # Test pickling w/ a subclass of ndarray
        a = uncertain_array(np.matrix(list(range(10))), [1, 0, 1, 0, 0] * 2)
        a_pickled = pickle.loads(a.dumps())
        assert_equal(a_pickled._mask, a._mask)
        assert_equal(a_pickled, a)
        self.assertTrue(isinstance(a_pickled._data, np.matrix))

    def test_pickling_uncertainconstant(self):
        # Test pickling UncertainConstant
        mc = np.ma.uncertain
        mc_pickled = pickle.loads(mc.dumps())
        assert_equal(mc_pickled._baseclass, mc._baseclass)
        assert_equal(mc_pickled._mask, mc._mask)
        assert_equal(mc_pickled._data, mc._data)

    def test_pickling_keepalignment(self):
        # Tests pickling w/ F_CONTIGUOUS arrays
        a = np.arange(10)
        a.shape = (-1, 2)
        b = a.T
        test = pickle.loads(pickle.dumps(b))
        assert_equal(test, b)

    def test_single_element_subscript(self):
        # Tests single element subscripts of Uncertainarrays.
        a = uncertain_array([1, 3, 2])
        b = uncertain_array([1, 3, 2], [1, 0, 1])
        assert_equal(a[0].shape, ())
        assert_equal(b[0].shape, ())
        assert_equal(b[1].shape, ())

    def test_topython(self):
        # Tests some communication issues with Python.
        assert_equal(1, int(array(1)))
        assert_equal(1.0, float(array(1)))
        assert_equal(1, int(array([[[1]]])))
        assert_equal(1.0, float(array([[1]])))
        self.assertRaises(TypeError, float, array([1, 1]))

        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Warning: converting a uncertain element')
            assert_(np.isnan(float(array([1], [1]))))

            a = uncertain_array([1, 2, 3], [1, 0, 0])
            self.assertRaises(TypeError, lambda: float(a))
            assert_equal(float(a[-1]), 3.)
            self.assertTrue(np.isnan(float(a[0])))
        self.assertRaises(TypeError, int, a)
        assert_equal(int(a[-1]), 3)
        self.assertRaises(MAError, lambda:int(a[0]))

    def test_oddfeatures_1(self):
        # Test of other odd features
        x = np.arange(20)
        x = x.reshape(4, 5)
        x.flat[5] = 12
        assert_(x[1, 0] == 12)
        z = x + 10j * x
        assert_equal(z.real, x)
        assert_equal(z.imag, 10 * x)
        assert_equal((z * conjugate(z)).real, 101 * x * x)
        z.imag[...] = 0.0

        x = np.arange(10)
        x[3] = uncertain
        assert_(str(x[3]) == str(uncertain))
        c = x >= 8
        assert_(count(where(c, uncertain, uncertain)) == 0)
        assert_(shape(where(c, uncertain, uncertain)) == c.shape)

        z = uncertain_where(c, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is uncertain)
        assert_(z[4] is not uncertain)
        assert_(z[7] is not uncertain)
        assert_(z[8] is uncertain)
        assert_(z[9] is uncertain)
        assert_equal(x, z)

    def test_oddfeatures_2(self):
        # Tests some more features.
        x = uncertain_array([1., 2., 3., 4., 5.])
        c = uncertain_array([1, 1, 1, 0, 0])
        x[2] = uncertain
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        c[0] = uncertain
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        assert_(z[0] is uncertain)
        assert_(z[1] is not uncertain)
        assert_(z[2] is uncertain)

    def test_oddfeatures_3(self):
        # Tests some generic features
        atest = uncertain_array([10], True)
        btest = uncertain_array([20])
        idx = atest.mask
        atest[idx] = btest[idx]
        assert_equal(atest, [20])

    def test_optinfo_propagation(self):
        # Checks that _optinfo dictionary isn't back-propagated
        x = uncertain_array([1, 2, 3, ], dtype=float)
        x._optinfo['info'] = '???'
        y = x.copy()
        assert_equal(y._optinfo['info'], '???')
        y._optinfo['info'] = '!!!'
        assert_equal(x._optinfo['info'], '???')

    def test_fancy_printoptions(self):
        # Test printing a uncertain array w/ fancy dtype.
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = uncertain_array([(1, (2, 3.0)), (4, (5, 6.0))],
                     [(1, (0, 1)), (0, (1, 0))],
                     dtype=fancydtype)
        control = "[(--, (2, --)) (4, (--, 6.0))]"
        assert_equal(str(test), control)

        # Test 0-d array with multi-dimensional dtype
        t_2d0 = uncertain_array(data = (0, [[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0]],
                                    0.0),
                             mask = (False, [[True, False, True],
                                             [False, False, True]],
                                     False),
                             dtype = "int, (2,3)float, float")
        control = "(0, [[--, 0.0, --], [0.0, 0.0, --]], 0.0)"
        assert_equal(str(t_2d0), control)

    def test_object_with_array(self):
        mx1 = uncertain_array([1.], [True])
        mx2 = uncertain_array([1., 2.])
        mx = uncertain_array([mx1, mx2], [False, True])
        assert_(mx[0] is mx1)
        assert_(mx[1] is not mx2)
        assert_(np.all(mx[1].data == mx2.data))
        assert_(np.all(mx[1].mask))
        # check that we return a view.
        mx[1].data[0] = 0.
        assert_(mx2[0] == 0.)


class TestUncertainArrayArithmetic(TestCase):
    # Base test class for UncertainArrays.

    def setUp(self):
        # Base data definition.
        raise SkipTest
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = np.array([1, 0, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 0])
        m2 = np.array([0, 0, 3, 0, 0, 6, 1, 0, 0, 0, 0, 1])
        xm = uarray(x, m1, dtype='uarray')
        ym = uarray(y, m2, dtype='uarray')
        z = np.array([-.5, 0., .5, .8])
        zm = uarray(z, [0, 1, 0, 0], dtype='uarray')
        xf = np.where(m1, 1e+20, x)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)
        self.err_status = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')

    def tearDown(self):
        np.seterr(**self.err_status)

    def test_basic_arithmetic(self):
        # Test of basic arithmetic.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        a2d = uncertain_array([[1, 2], [0, 4]])
        a2dm = uncertain_array(a2d, [[0, 0], [1, 0]])
        assert_equal(a2d * a2d, a2d * a2dm)
        assert_equal(a2d + a2d, a2d + a2dm)
        assert_equal(a2d - a2d, a2d - a2dm)
        for s in [(12,), (4, 3), (2, 6)]:
            x = x.reshape(s)
            y = y.reshape(s)
            xm = xm.reshape(s)
            ym = ym.reshape(s)
            xf = xf.reshape(s)
            assert_equal(-x, -xm)
            assert_equal(x + y, xm + ym)
            assert_equal(x - y, xm - ym)
            assert_equal(x * y, xm * ym)
            assert_equal(x / y, xm / ym)
            assert_equal(a10 + y, a10 + ym)
            assert_equal(a10 - y, a10 - ym)
            assert_equal(a10 * y, a10 * ym)
            assert_equal(a10 / y, a10 / ym)
            assert_equal(x + a10, xm + a10)
            assert_equal(x - a10, xm - a10)
            assert_equal(x * a10, xm * a10)
            assert_equal(x / a10, xm / a10)
            assert_equal(x ** 2, xm ** 2)
            assert_equal(abs(x) ** 2.5, abs(xm) ** 2.5)
            assert_equal(x ** y, xm ** ym)
            assert_equal(np.add(x, y), add(xm, ym))
            assert_equal(np.subtract(x, y), subtract(xm, ym))
            assert_equal(np.multiply(x, y), multiply(xm, ym))
            assert_equal(np.divide(x, y), divide(xm, ym))

    def test_divide_on_different_shapes(self):
        x = np.arange(6, dtype=float)
        x.shape = (2, 3)
        y = np.arange(3, dtype=float)

        z = x / y
        assert_equal(z, [[-1., 1., 1.], [-1., 4., 2.5]])
        assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])

        z = x / y[None,:]
        assert_equal(z, [[-1., 1., 1.], [-1., 4., 2.5]])
        assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])

        y = np.arange(2, dtype=float)
        z = x / y[:, None]
        assert_equal(z, [[-1., -1., -1.], [3., 4., 5.]])
        assert_equal(z.mask, [[1, 1, 1], [0, 0, 0]])

    def test_mixed_arithmetic(self):
        # Tests mixed arithmetics.
        na = np.array([1])
        ma = uncertain_array([1])
        self.assertTrue(isinstance(na + ma, UncertainArray))
        self.assertTrue(isinstance(ma + na, UncertainArray))

    def test_limits_arithmetic(self):
        tiny = np.finfo(float).tiny
        a = uncertain_array([tiny, 1. / tiny, 0.])
        assert_equal(getmaskarray(a / 2), [0, 0, 0])
        assert_equal(getmaskarray(2 / a), [1, 0, 1])

    def test_uncertain_singleton_arithmetic(self):
        # Tests some scalar arithmetics on UncertainArrays.
        # Uncertain singleton should remain uncertain no matter what
        xm = uncertain_array(0, 1)
        self.assertTrue((1 / array(0)).mask)
        self.assertTrue((1 + xm).mask)
        self.assertTrue((-xm).mask)
        self.assertTrue(maximum(xm, xm).mask)
        self.assertTrue(minimum(xm, xm).mask)

    def test_uncertain_singleton_equality(self):
        # Tests (in)equality on uncertain singleton
        a = uncertain_array([1, 2, 3], [1, 1, 0])
        assert_((a[0] == 0) is uncertain)
        assert_((a[0] != 0) is uncertain)
        assert_equal((a[-1] == 0), False)
        assert_equal((a[-1] != 0), True)

    def test_arithmetic_with_uncertain_singleton(self):
        # Checks that there's no collapsing to uncertain
        x = uncertain_array([1, 2])
        y = x * uncertain
        assert_equal(y.shape, x.shape)
        assert_equal(y._mask, [True, True])
        y = x[0] * uncertain
        assert_(y is uncertain)
        y = x + uncertain
        assert_equal(y.shape, x.shape)
        assert_equal(y._mask, [True, True])

    def test_arithmetic_with_uncertain_singleton_on_1d_singleton(self):
        # Check that we're not losing the shape of a singleton
        x = uncertain_array([1, ])
        y = x + uncertain
        assert_equal(y.shape, x.shape)
        assert_equal(y.mask, [True, ])

    def test_scalar_arithmetic(self):
        x = uncertain_array(0, 0)
        assert_equal(x.filled().ctypes.data, x.ctypes.data)
        # Make sure we don't lose the shape in some circumstances
        xm = uncertain_array((0, 0)) / 0.
        assert_equal(xm.shape, (2,))
        assert_equal(xm.mask, [1, 1])

    def test_basic_ufuncs(self):
        # Test various functions such as sin, cos.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_equal(np.cos(x), cos(xm))
        assert_equal(np.cosh(x), cosh(xm))
        assert_equal(np.sin(x), sin(xm))
        assert_equal(np.sinh(x), sinh(xm))
        assert_equal(np.tan(x), tan(xm))
        assert_equal(np.tanh(x), tanh(xm))
        assert_equal(np.sqrt(abs(x)), sqrt(xm))
        assert_equal(np.log(abs(x)), log(xm))
        assert_equal(np.log10(abs(x)), log10(xm))
        assert_equal(np.exp(x), exp(xm))
        assert_equal(np.arcsin(z), arcsin(zm))
        assert_equal(np.arccos(z), arccos(zm))
        assert_equal(np.arctan(z), arctan(zm))
        assert_equal(np.arctan2(x, y), arctan2(xm, ym))
        assert_equal(np.absolute(x), absolute(xm))
        assert_equal(np.angle(x + 1j*y), angle(xm + 1j*ym))
        assert_equal(np.angle(x + 1j*y, deg=True), angle(xm + 1j*ym, deg=True))
        assert_equal(np.equal(x, y), equal(xm, ym))
        assert_equal(np.not_equal(x, y), not_equal(xm, ym))
        assert_equal(np.less(x, y), less(xm, ym))
        assert_equal(np.greater(x, y), greater(xm, ym))
        assert_equal(np.less_equal(x, y), less_equal(xm, ym))
        assert_equal(np.greater_equal(x, y), greater_equal(xm, ym))
        assert_equal(np.conjugate(x), conjugate(xm))

    def test_count_func(self):
        # Tests count
        assert_equal(1, count(1))
        assert_equal(0, array(1, [1]))

        ott = uncertain_array([0., 1., 2., 3.], [1, 0, 0, 0])
        res = count(ott)
        self.assertTrue(res.dtype.type is np.intp)
        assert_equal(3, res)

        ott = ott.reshape((2, 2))
        res = count(ott)
        assert_(res.dtype.type is np.intp)
        assert_equal(3, res)
        res = count(ott, 0)
        assert_(isinstance(res, ndarray))
        assert_equal([1, 2], res)
        assert_(getmask(res) is nomask)

        ott = uncertain_array([0., 1., 2., 3.])
        res = count(ott, 0)
        assert_(isinstance(res, ndarray))
        assert_(res.dtype.type is np.intp)
        assert_raises(ValueError, ott.count, axis=1)

    def test_minmax_func(self):
        # Tests minimum and maximum.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # max doesn't work if shaped
        xr = np.ravel(x)
        xmr = ravel(xm)
        # following are true because of careful selection of data
        assert_equal(max(xr), maximum(xmr))
        assert_equal(min(xr), minimum(xmr))

        assert_equal(minimum([1, 2, 3], [4, 0, 9]), [1, 0, 3])
        assert_equal(maximum([1, 2, 3], [4, 0, 9]), [4, 2, 9])
        x = np.arange(5)
        y = np.arange(5) - 2
        x[3] = uncertain
        y[0] = uncertain
        assert_equal(minimum(x, y), where(less(x, y), x, y))
        assert_equal(maximum(x, y), where(greater(x, y), x, y))
        assert_(minimum(x) == 0)
        assert_(maximum(x) == 4)

        x = np.arange(4).reshape(2, 2)
        x[-1, -1] = uncertain
        assert_equal(maximum(x), 2)

    def test_minimummaximum_func(self):
        a = np.ones((2, 2))
        aminimum = minimum(a, a)
        self.assertTrue(isinstance(aminimum, UncertainArray))
        assert_equal(aminimum, np.minimum(a, a))

        aminimum = minimum.outer(a, a)
        self.assertTrue(isinstance(aminimum, UncertainArray))
        assert_equal(aminimum, np.minimum.outer(a, a))

        amaximum = maximum(a, a)
        self.assertTrue(isinstance(amaximum, UncertainArray))
        assert_equal(amaximum, np.maximum(a, a))

        amaximum = maximum.outer(a, a)
        self.assertTrue(isinstance(amaximum, UncertainArray))
        assert_equal(amaximum, np.maximum.outer(a, a))

    def test_minmax_reduce(self):
        # Test np.min/maximum.reduce on array w/ full False mask
        a = uncertain_array([1, 2, 3], [False, False, False])
        b = np.maximum.reduce(a)
        assert_equal(b, 3)

    def test_minmax_funcs_with_output(self):
        # Tests the min/max functions with explicit outputs
        mask = np.random.rand(12).round()
        xm = uncertain_array(np.random.uniform(0, 10, 12), mask)
        xm.shape = (3, 4)
        for funcname in ('min', 'max'):
            # Initialize
            npfunc = getattr(np, funcname)
            mafunc = getattr(uncertainties.unumpy.core2, funcname)
            # Use the np version
            nout = np.empty((4,), dtype=int)
            try:
                result = npfunc(xm, axis=0, out=nout)
            except MaskError:
                pass
            nout = np.empty((4,), dtype=float)
            result = npfunc(xm, axis=0, out=nout)
            self.assertTrue(result is nout)
            # Use the ma version
            nout.fill(-999)
            result = mafunc(xm, axis=0, out=nout)
            self.assertTrue(result is nout)

    def test_minmax_methods(self):
        # Additional tests on max/min
        (_, _, _, _, _, xm, _, _, _, _) = self.d
        xm.shape = (xm.size,)
        assert_equal(xm.max(), 10)
        self.assertTrue(xm[0].max() is uncertain)
        self.assertTrue(xm[0].max(0) is uncertain)
        self.assertTrue(xm[0].max(-1) is uncertain)
        assert_equal(xm.min(), -10.)
        self.assertTrue(xm[0].min() is uncertain)
        self.assertTrue(xm[0].min(0) is uncertain)
        self.assertTrue(xm[0].min(-1) is uncertain)
        assert_equal(xm.ptp(), 20.)
        self.assertTrue(xm[0].ptp() is uncertain)
        self.assertTrue(xm[0].ptp(0) is uncertain)
        self.assertTrue(xm[0].ptp(-1) is uncertain)

        x = uncertain_array([1, 2, 3])
        self.assertTrue(x.min() is uncertain)
        self.assertTrue(x.max() is uncertain)
        self.assertTrue(x.ptp() is uncertain)

    def test_addsumprod(self):
        # Tests add, sum, product.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_equal(np.add.reduce(x), add.reduce(x))
        assert_equal(np.add.accumulate(x), add.accumulate(x))
        assert_equal(4, sum(array(4), axis=0))
        assert_equal(4, sum(array(4), axis=0))
        assert_equal(np.sum(x, axis=0), sum(x, axis=0))
        assert_equal(np.sum(filled(xm, 0), axis=0), sum(xm, axis=0))
        assert_equal(np.sum(x, 0), sum(x, 0))
        assert_equal(np.product(x, axis=0), product(x, axis=0))
        assert_equal(np.product(x, 0), product(x, 0))
        assert_equal(np.product(filled(xm, 1), axis=0), product(xm, axis=0))
        s = (3, 4)
        x.shape = y.shape = xm.shape = ym.shape = s
        if len(s) > 1:
            assert_equal(np.concatenate((x, y), 1), concatenate((xm, ym), 1))
            assert_equal(np.add.reduce(x, 1), add.reduce(x, 1))
            assert_equal(np.sum(x, 1), sum(x, 1))
            assert_equal(np.product(x, 1), product(x, 1))

    def test_binops_d2D(self):
        # Test binary operations on 2D data
        a = uncertain_array([[1.], [2.], [3.]], [[False], [True], [True]])
        b = uncertain_array([[2., 3.], [4., 5.], [6., 7.]])

        test = a * b
        control = uncertain_array([[2., 3.], [2., 2.], [3., 3.]],
                        [[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        test = b * a
        control = uncertain_array([[2., 3.], [4., 5.], [6., 7.]],
                        [[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        a = uncertain_array([[1.], [2.], [3.]])
        b = uncertain_array([[2., 3.], [4., 5.], [6., 7.]],
                  [[0, 0], [0, 0], [0, 1]])
        test = a * b
        control = uncertain_array([[2, 3], [8, 10], [18, 3]],
                        [[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        test = b * a
        control = uncertain_array([[2, 3], [8, 10], [18, 7]],
                        [[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

    def test_domained_binops_d2D(self):
        # Test domained binary operations on 2D data
        a = uncertain_array([[1.], [2.], [3.]], [[False], [True], [True]])
        b = uncertain_array([[2., 3.], [4., 5.], [6., 7.]])

        test = a / b
        control = uncertain_array([[1. / 2., 1. / 3.], [2., 2.], [3., 3.]],
                        [[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        test = b / a
        control = uncertain_array([[2. / 1., 3. / 1.], [4., 5.], [6., 7.]],
                        [[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        a = uncertain_array([[1.], [2.], [3.]])
        b = uncertain_array([[2., 3.], [4., 5.], [6., 7.]],
                  [[0, 0], [0, 0], [0, 1]])
        test = a / b
        control = uncertain_array([[1. / 2, 1. / 3], [2. / 4, 2. / 5], [3. / 6, 3]],
                        [[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        test = b / a
        control = uncertain_array([[2 / 1., 3 / 1.], [4 / 2., 5 / 2.], [6 / 3., 7]],
                        [[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

    def test_ufunc_nomask(self):
        # check the case ufuncs should set the mask to false
        m = uarray([1], dtype='uarray')
        # check we don't get array([False], dtype=bool)
        assert_equal(np.true_divide(m, 5).mask.shape, ())

    def test_mod(self):
        # Tests mod
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_equal(mod(x, y), mod(xm, ym))
        test = mod(ym, xm)
        assert_equal(test, np.mod(ym, xm))
        assert_equal(test.mask, mask_or(xm.mask, ym.mask))
        test = mod(xm, ym)
        assert_equal(test, np.mod(xm, ym))
        assert_equal(test.mask, mask_or(mask_or(xm.mask, ym.mask), (ym == 0)))

    def test_TakeTransposeInnerOuter(self):
        # Test of take, transpose, inner, outer products
        x = np.arange(24)
        y = np.arange(24)
        x[5:6] = uncertain
        x = x.reshape(2, 3, 4)
        y = y.reshape(2, 3, 4)
        assert_equal(np.transpose(y, (2, 0, 1)), transpose(x, (2, 0, 1)))
        assert_equal(np.take(y, (2, 0, 1), 1), take(x, (2, 0, 1), 1))
        assert_equal(np.inner(filled(x, 0), filled(y, 0)),
                     inner(x, y))
        assert_equal(np.outer(filled(x, 0), filled(y, 0)),
                     outer(x, y))
        y = uncertain_array(['abc', 1, 'def', 2, 3], object)
        y[2] = uncertain
        t = take(y, [0, 3, 4])
        assert_(t[0] == 'abc')
        assert_(t[1] == 2)
        assert_(t[2] == 3)

    def test_imag_real(self):
        # Check complex
        xx = uncertain_array([1 + 10j, 20 + 2j], [1, 0])
        assert_equal(xx.imag, [10, 2])
        assert_equal(xx.imag.filled(), [1e+20, 2])
        assert_equal(xx.imag.dtype, xx._data.imag.dtype)
        assert_equal(xx.real, [1, 20])
        assert_equal(xx.real.filled(), [1e+20, 20])
        assert_equal(xx.real.dtype, xx._data.real.dtype)

    def test_methods_with_output(self):
        xm = uncertain_array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = uncertain

        funclist = ('sum', 'prod', 'var', 'std', 'max', 'min', 'ptp', 'mean',)

        for funcname in funclist:
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)
            # A ndarray as explicit input
            output = np.empty(4, dtype=float)
            output.fill(-9999)
            result = npfunc(xm, axis=0, out=output)
            # ... the result should be the given output
            assert_(result is output)
            assert_equal(result, xmmeth(axis=0, out=output))

            output = empty(4, dtype=int)
            result = xmmeth(axis=0, out=output)
            assert_(result is output)
            assert_(output[0] is uncertain)

    def test_count_mean_with_matrix(self):
        m = uarray(np.matrix([[1,2],[3,4]]), np.zeros((2,2)), dtype='uarray')

        assert_equal(m.count(axis=0).shape, (1,2))
        assert_equal(m.count(axis=1).shape, (2,1))

        #make sure broadcasting inside mean and var work
        assert_equal(m.mean(axis=0), [[2., 3.]])
        assert_equal(m.mean(axis=1), [[1.5], [3.5]])

    def test_eq_with_None(self):
        # Really, comparisons with None should not be done, but check them
        # anyway. Note that pep8 will flag these tests.
        # Deprecation is in place for arrays, and when it happens this
        # test will fail (and have to be changed accordingly).

        # With partial mask
        with suppress_warnings() as sup:
            sup.filter(FutureWarning, "Comparison to `None`")
            a = uncertain_array([1, 2], [0, 1])
            assert_equal(a == None, False)
            assert_equal(a.data == None, False)
            assert_equal(a.mask == None, False)
            assert_equal(a != None, True)
            # With nomask
            a = uncertain_array([1, 2] )
            assert_equal(a == None, False)
            assert_equal(a != None, True)
            # With complete mask
            a = uncertain_array([1, 2])
            assert_equal(a == None, False)
            assert_equal(a != None, True)
            # Fully uncertain, even comparison to None should return "uncertain"
            a = uncertain
            assert_equal(a == None, uncertain)

    def test_eq_with_scalar(self):
        a = uncertain_array(1)
        assert_equal(a == 1, True)
        assert_equal(a == 0, False)
        assert_equal(a != 1, False)
        assert_equal(a != 0, True)

    def test_numpyarithmetics(self):
        # Check that the mask is not back-propagated when using numpy functions
        a = uncertain_array([-1, 0, 1, 2, 3], [0, 0, 0, 0, 1])
        control = uncertain_array([np.nan, np.nan, 0, np.log(2), -1],
                               [1, 1, 0, 0, 1])

        test = log(a)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(a.mask, [0, 0, 0, 0, 1])

        test = np.log(a)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(a.mask, [0, 0, 0, 0, 1])


class TestUncertainArrayAttributes(TestCase):

    def test_flat(self):
        # Test that flat can return all types of items [#4585, #4615]
        # test simple access
        test = uncertain_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        assert_equal(test.flat[1], 2)
        assert_equal(test.flat[2], uncertain)
        self.assertTrue(np.all(test.flat[0:2] == test[0, 0:2]))
        # Test flat on uncertain_matrices
        test = uncertain_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        test.flat = uncertain_array([3, 2, 1], mask=[1, 0, 0])
        control = uncertain_array(np.matrix([[3, 2, 1]]), mask=[1, 0, 0])
        assert_equal(test, control)
        # Test setting
        test = uncertain_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        testflat = test.flat
        testflat[:] = testflat[[2, 1, 0]]
        assert_equal(test, control)
        testflat[0] = 9
        assert_equal(test[0, 0], 9)
        # test 2-D record array
        # ... on structured array w/ uncertain records
        x = uncertain_array([[(1, 1.1, 'one'), (2, 2.2, 'two'), (3, 3.3, 'thr')],
                   [(4, 4.4, 'fou'), (5, 5.5, 'fiv'), (6, 6.6, 'six')]],
                  dtype=[('a', int), ('b', float), ('c', '|S8')])
        x['a'][0, 1] = uncertain
        x['b'][1, 0] = uncertain
        x['c'][0, 2] = uncertain
        x[-1, -1] = uncertain
        xflat = x.flat
        assert_equal(xflat[0], x[0, 0])
        assert_equal(xflat[1], x[0, 1])
        assert_equal(xflat[2], x[0, 2])
        assert_equal(xflat[:3], x[0])
        assert_equal(xflat[3], x[1, 0])
        assert_equal(xflat[4], x[1, 1])
        assert_equal(xflat[5], x[1, 2])
        assert_equal(xflat[3:], x[1])
        assert_equal(xflat[-1], x[-1, -1])
        i = 0
        j = 0
        for xf in xflat:
            assert_equal(xf, x[j, i])
            i += 1
            if i >= x.shape[-1]:
                i = 0
                j += 1
        # test that matrices keep the correct shape (#4615)
        a = uncertain_array(np.matrix(np.eye(2)), mask=0)
        b = a.flat
        b01 = b[:2]
        assert_equal(b01.data, array([[1., 0.]]))
        assert_equal(b01.mask, array([[False, False]]))

    def test_assign_dtype(self):
        # check that the mask's dtype is updated when dtype is changed
        a = np.zeros(4, dtype='f4,i4')

        m = uarray(a, dtype='uarray')
        m.dtype = np.dtype('f4')
        repr(m)  # raises?
        assert_equal(m.dtype, np.dtype('f4'))

        # check that dtype changes that change shape of mask too much
        # are not allowed
        def assign():
            m = uarray(a, dtype='uarray')
            m.dtype = np.dtype('f8')
        assert_raises(ValueError, assign)

        b = a.view(dtype='f4', type=np.ma.UncertainArray)  # raises?
        assert_equal(b.dtype, np.dtype('f4'))

        # check that nomask is preserved
        a = np.zeros(4, dtype='f4')
        m = np.ma.array(a)
        m.dtype = np.dtype('f4,i4')
        assert_equal(m.dtype, np.dtype('f4,i4'))
        assert_equal(m._mask, np.ma.nomask)


class TestUfuncs(TestCase):
    # Test class for the application of ufuncs on UncertainArrays.

    def setUp(self):
        # Base data definition.
        raise SkipTest
        self.d = (array([1.0, 0, -1, pi / 2] * 2, mask=[0, 1] + [0] * 6),
                  array([1.0, 0, -1, pi / 2] * 2, mask=[1, 0] + [0] * 6),)
        self.err_status = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')

    def tearDown(self):
        np.seterr(**self.err_status)

    def test_testUfuncRegression(self):
        # Tests new ufuncs on UncertainArrays.
        for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate',
                  'sin', 'cos', 'tan',
                  'arcsin', 'arccos', 'arctan',
                  'sinh', 'cosh', 'tanh',
                  'arcsinh',
                  'arccosh',
                  'arctanh',
                  'absolute', 'fabs', 'negative',
                  'floor', 'ceil',
                  'logical_not',
                  'add', 'subtract', 'multiply',
                  'divide', 'true_divide', 'floor_divide',
                  'remainder', 'fmod', 'hypot', 'arctan2',
                  'equal', 'not_equal', 'less_equal', 'greater_equal',
                  'less', 'greater',
                  'logical_and', 'logical_or', 'logical_xor',
                  ]:
            try:
                uf = getattr(umath, f)
            except AttributeError:
                uf = getattr(fromnumeric, f)
            mf = getattr(uncertainties.unumpy.core2, f)
            args = self.d[:uf.nin]
            ur = uf(*args)
            mr = mf(*args)
            assert_equal(ur.filled(0), mr.filled(0), f)
            assert_mask_equal(ur.mask, mr.mask, err_msg=f)

    def test_reduce(self):
        # Tests reduce on UncertainArrays.
        a = self.d[0]
        self.assertTrue(not alltrue(a, axis=0))
        self.assertTrue(sometrue(a, axis=0))
        assert_equal(sum(a[:3], axis=0), 0)
        assert_equal(product(a, axis=0), 0)
        assert_equal(add.reduce(a), pi)

    def test_minmax(self):
        # Tests extrema on UncertainArrays.
        a = np.arange(1, 13).reshape(3, 4)
        amask = uncertain_where(a < 5, a)
        assert_equal(amask.max(), a.max())
        assert_equal(amask.min(), 5)
        assert_equal(amask.max(0), a.max(0))
        assert_equal(amask.min(0), [5, 6, 7, 8])
        self.assertTrue(amask.max(1)[0].mask)
        self.assertTrue(amask.min(1)[0].mask)

    def test_ndarray_mask(self):
        # Check that the mask of the result is a ndarray (not a UncertainArray...)
        a = uncertain_array([-1, 0, 1, 2, 3], mask=[0, 0, 0, 0, 1])
        test = np.sqrt(a)
        control = uncertain_array([-1, 0, 1, np.sqrt(2), -1],
                               mask=[1, 0, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        self.assertTrue(not isinstance(test.mask, UncertainArray))

    def test_treatment_of_NotImplemented(self):
        # Check that NotImplemented is returned at appropriate places

        a = uncertain_array([1., 2.], mask=[1, 0])
        self.assertRaises(TypeError, operator.mul, a, "abc")
        self.assertRaises(TypeError, operator.truediv, a, "abc")

        class MyClass(object):
            __array_priority__ = a.__array_priority__ + 1

            def __mul__(self, other):
                return "My mul"

            def __rmul__(self, other):
                return "My rmul"

        me = MyClass()
        assert_(me * a == "My mul")
        assert_(a * me == "My rmul")

        # and that __array_priority__ is respected
        class MyClass2(object):
            __array_priority__ = 100

            def __mul__(self, other):
                return "Me2mul"

            def __rmul__(self, other):
                return "Me2rmul"

            def __rdiv__(self, other):
                return "Me2rdiv"

            __rtruediv__ = __rdiv__

        me_too = MyClass2()
        assert_(a.__mul__(me_too) is NotImplemented)
        assert_(all(multiply.outer(a, me_too) == "Me2rmul"))
        assert_(a.__truediv__(me_too) is NotImplemented)
        assert_(me_too * a == "Me2mul")
        assert_(a * me_too == "Me2rmul")
        assert_(a / me_too == "Me2rdiv")

    def test_no_uncertain_nan_warnings(self):
        # check that a nan in uncertain position does not
        # cause ufunc warnings

        m = np.ma.array([0.5, np.nan], mask=[0,1])

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            # test unary and binary ufuncs
            exp(m)
            add(m, 1)
            m > 0

            # test different unary domains
            sqrt(m)
            log(m)
            tan(m)
            arcsin(m)
            arccos(m)
            arccosh(m)

            # test binary domains
            divide(m, 2)

            # also check that allclose uses ma ufuncs, to avoid warning
            allclose(m, 0.5)

class TestUncertainArrayInPlaceArithmetics(TestCase):
    # Test UncertainArray Arithmetics

    def setUp(self):
        raise SkipTest
        x = np.arange(10)
        y = np.arange(10)
        xm = np.arange(10)
        self.intdata = (x, y, xm)
        self.floatdata = (x.astype(float), y.astype(float), xm.astype(float))
        self.othertypes = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        self.othertypes = [np.dtype(_).type for _ in self.othertypes]
        self.uint8data = (
            x.astype(np.uint8),
            y.astype(np.uint8),
            xm.astype(np.uint8)
        )

    def test_inplace_addition_scalar(self):
        # Test of inplace additions
        (x, y, xm) = self.intdata
        xm[2] = uncertain
        x += 1
        assert_equal(x, y + 1)
        xm += 1
        assert_equal(xm, y + 1)

        (x, _, xm) = self.floatdata
        id1 = x.data.ctypes._data
        x += 1.
        assert_(id1 == x.data.ctypes._data)
        assert_equal(x, y + 1.)

    def test_inplace_addition_array(self):
        # Test of inplace additions
        (x, y, xm) = self.intdata
        m = xm.mask
        a = np.arange(10, dtype=np.int16)
        a[-1] = uncertain
        x += a
        xm += a
        assert_equal(x, y + a)
        assert_equal(xm, y + a)
        assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_subtraction_scalar(self):
        # Test of inplace subtractions
        (x, y, xm) = self.intdata
        x -= 1
        assert_equal(x, y - 1)
        xm -= 1
        assert_equal(xm, y - 1)

    def test_inplace_subtraction_array(self):
        # Test of inplace subtractions
        (x, y, xm) = self.floatdata
        m = xm.mask
        a = np.arange(10, dtype=float)
        a[-1] = uncertain
        x -= a
        xm -= a
        assert_equal(x, y - a)
        assert_equal(xm, y - a)
        assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_multiplication_scalar(self):
        # Test of inplace multiplication
        (x, y, xm) = self.floatdata
        x *= 2.0
        assert_equal(x, y * 2)
        xm *= 2.0
        assert_equal(xm, y * 2)

    def test_inplace_multiplication_array(self):
        # Test of inplace multiplication
        (x, y, xm) = self.floatdata
        m = xm.mask
        a = np.arange(10, dtype=float)
        a[-1] = uncertain
        x *= a
        xm *= a
        assert_equal(x, y * a)
        assert_equal(xm, y * a)
        assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_division_scalar_int(self):
        # Test of inplace division
        (x, y, xm) = self.intdata
        x = np.arange(10) * 2
        xm = np.arange(10) * 2
        xm[2] = uncertain
        x //= 2
        assert_equal(x, y)
        xm //= 2
        assert_equal(xm, y)

    def test_inplace_division_scalar_float(self):
        # Test of inplace division
        (x, y, xm) = self.floatdata
        x /= 2.0
        assert_equal(x, y / 2.0)
        xm /= np.arange(10)
        assert_equal(xm, ones((10,)))

    def test_inplace_division_array_float(self):
        # Test of inplace division
        (x, y, xm) = self.floatdata
        m = xm.mask
        a = np.arange(10, dtype=float)
        a[-1] = uncertain
        x /= a
        xm /= a
        assert_equal(x, y / a)
        assert_equal(xm, y / a)
        assert_equal(xm.mask, mask_or(mask_or(m, a.mask), (a == 0)))

    def test_inplace_division_misc(self):

        x = [1., 1., 1., -2., pi / 2., 4., 5., -10., 10., 1., 2., 3.]
        y = [5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.]
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = uncertain_array(x, mask=m1)
        ym = uncertain_array(y, mask=m2)

        z = xm / ym
        assert_equal(z._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        assert_equal(z._data,
                     [1., 1., 1., -1., -pi / 2., 4., 5., 1., 1., 1., 2., 3.])

        xm = xm.copy()
        xm /= ym
        assert_equal(xm._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        assert_equal(z._data,
                     [1., 1., 1., -1., -pi / 2., 4., 5., 1., 1., 1., 2., 3.])

    def test_datafriendly_add(self):
        # Test keeping data w/ (inplace) addition
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        # Test add w/ scalar
        xx = x + 1
        assert_equal(xx.data, [2, 3, 3])
        assert_equal(xx.mask, [0, 0, 1])
        # Test iadd w/ scalar
        x += 1
        assert_equal(x.data, [2, 3, 3])
        assert_equal(x.mask, [0, 0, 1])
        # Test add w/ array
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        xx = x + array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 4, 3])
        assert_equal(xx.mask, [1, 0, 1])
        # Test iadd w/ array
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        x += uncertain_array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(x.data, [1, 4, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_sub(self):
        # Test keeping data w/ (inplace) subtraction
        # Test sub w/ scalar
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        xx = x - 1
        assert_equal(xx.data, [0, 1, 3])
        assert_equal(xx.mask, [0, 0, 1])
        # Test isub w/ scalar
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        x -= 1
        assert_equal(x.data, [0, 1, 3])
        assert_equal(x.mask, [0, 0, 1])
        # Test sub w/ array
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        xx = x - array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 0, 3])
        assert_equal(xx.mask, [1, 0, 1])
        # Test isub w/ array
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        x -= uncertain_array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(x.data, [1, 0, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_mul(self):
        # Test keeping data w/ (inplace) multiplication
        # Test mul w/ scalar
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        xx = x * 2
        assert_equal(xx.data, [2, 4, 3])
        assert_equal(xx.mask, [0, 0, 1])
        # Test imul w/ scalar
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        x *= 2
        assert_equal(x.data, [2, 4, 3])
        assert_equal(x.mask, [0, 0, 1])
        # Test mul w/ array
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        xx = x * array([10, 20, 30], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 40, 3])
        assert_equal(xx.mask, [1, 0, 1])
        # Test imul w/ array
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        x *= uncertain_array([10, 20, 30], mask=[1, 0, 0])
        assert_equal(x.data, [1, 40, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_div(self):
        # Test keeping data w/ (inplace) division
        # Test div on scalar
        x = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        xx = x / 2.
        assert_equal(xx.data, [1 / 2., 2 / 2., 3])
        assert_equal(xx.mask, [0, 0, 1])
        # Test idiv on scalar
        x = uncertain_array([1., 2., 3.], mask=[0, 0, 1])
        x /= 2.
        assert_equal(x.data, [1 / 2., 2 / 2., 3])
        assert_equal(x.mask, [0, 0, 1])
        # Test div on array
        x = uncertain_array([1., 2., 3.], mask=[0, 0, 1])
        xx = x / array([10., 20., 30.], mask=[1, 0, 0])
        assert_equal(xx.data, [1., 2. / 20., 3.])
        assert_equal(xx.mask, [1, 0, 1])
        # Test idiv on array
        x = uncertain_array([1., 2., 3.], mask=[0, 0, 1])
        x /= uncertain_array([10., 20., 30.], mask=[1, 0, 0])
        assert_equal(x.data, [1., 2 / 20., 3.])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_pow(self):
        # Test keeping data w/ (inplace) power
        # Test pow on scalar
        x = uncertain_array([1., 2., 3.], mask=[0, 0, 1])
        xx = x ** 2.5
        assert_equal(xx.data, [1., 2. ** 2.5, 3.])
        assert_equal(xx.mask, [0, 0, 1])
        # Test ipow on scalar
        x **= 2.5
        assert_equal(x.data, [1., 2. ** 2.5, 3])
        assert_equal(x.mask, [0, 0, 1])

    def test_datafriendly_add_arrays(self):
        a = uncertain_array([[1, 1], [3, 3]])
        b = uncertain_array([1, 1], mask=[0, 0])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])

        a = uncertain_array([[1, 1], [3, 3]])
        b = uncertain_array([1, 1], mask=[0, 1])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_datafriendly_sub_arrays(self):
        a = uncertain_array([[1, 1], [3, 3]])
        b = uncertain_array([1, 1], mask=[0, 0])
        a -= b
        assert_equal(a, [[0, 0], [2, 2]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])

        a = uncertain_array([[1, 1], [3, 3]])
        b = uncertain_array([1, 1], mask=[0, 1])
        a -= b
        assert_equal(a, [[0, 0], [2, 2]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_datafriendly_mul_arrays(self):
        a = uncertain_array([[1, 1], [3, 3]])
        b = uncertain_array([1, 1], mask=[0, 0])
        a *= b
        assert_equal(a, [[1, 1], [3, 3]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])

        a = uncertain_array([[1, 1], [3, 3]])
        b = uncertain_array([1, 1], mask=[0, 1])
        a *= b
        assert_equal(a, [[1, 1], [3, 3]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_inplace_addition_scalar_type(self):
        # Test of inplace additions
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                xm[2] = uncertain
                x += t(1)
                assert_equal(x, y + t(1))
                xm += t(1)
                assert_equal(xm, y + t(1))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_addition_array_type(self):
        # Test of inplace additions
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = np.arange(10, dtype=t)
                a[-1] = uncertain
                x += a
                xm += a
                assert_equal(x, y + a)
                assert_equal(xm, y + a)
                assert_equal(xm.mask, mask_or(m, a.mask))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_subtraction_scalar_type(self):
        # Test of inplace subtractions
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x -= t(1)
                assert_equal(x, y - t(1))
                xm -= t(1)
                assert_equal(xm, y - t(1))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_subtraction_array_type(self):
        # Test of inplace subtractions
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = np.arange(10, dtype=t)
                a[-1] = uncertain
                x -= a
                xm -= a
                assert_equal(x, y - a)
                assert_equal(xm, y - a)
                assert_equal(xm.mask, mask_or(m, a.mask))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_multiplication_scalar_type(self):
        # Test of inplace multiplication
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x *= t(2)
                assert_equal(x, y * t(2))
                xm *= t(2)
                assert_equal(xm, y * t(2))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_multiplication_array_type(self):
        # Test of inplace multiplication
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = np.arange(10, dtype=t)
                a[-1] = uncertain
                x *= a
                xm *= a
                assert_equal(x, y * a)
                assert_equal(xm, y * a)
                assert_equal(xm.mask, mask_or(m, a.mask))

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_floor_division_scalar_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x = np.arange(10, dtype=t) * t(2)
                xm = np.arange(10, dtype=t) * t(2)
                xm[2] = uncertain
                x //= t(2)
                xm //= t(2)
                assert_equal(x, y)
                assert_equal(xm, y)

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_floor_division_array_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = np.arange(10, dtype=t)
                a[-1] = uncertain
                x //= a
                xm //= a
                assert_equal(x, y // a)
                assert_equal(xm, y // a)
                assert_equal(
                    xm.mask,
                    mask_or(mask_or(m, a.mask), (a == t(0)))
                )

                assert_equal(len(w), 0, "Failed on type=%s." % t)

    def test_inplace_division_scalar_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with suppress_warnings() as sup:
                sup.record(UserWarning)

                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x = np.arange(10, dtype=t) * t(2)
                xm = np.arange(10, dtype=t) * t(2)
                xm[2] = uncertain

                # May get a DeprecationWarning or a TypeError.
                #
                # This is a consequence of the fact that this is true divide
                # and will require casting to float for calculation and
                # casting back to the original type. This will only be raised
                # with integers. Whether it is an error or warning is only
                # dependent on how stringent the casting rules are.
                #
                # Will handle the same way.
                try:
                    x /= t(2)
                    assert_equal(x, y)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= t(2)
                    assert_equal(xm, y)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)

                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, "Failed on type=%s." % t)
                else:
                    assert_equal(len(sup.log), 0, "Failed on type=%s." % t)

    def test_inplace_division_array_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with suppress_warnings() as sup:
                sup.record(UserWarning)
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = np.arange(10, dtype=t)
                a[-1] = uncertain

                # May get a DeprecationWarning or a TypeError.
                #
                # This is a consequence of the fact that this is true divide
                # and will require casting to float for calculation and
                # casting back to the original type. This will only be raised
                # with integers. Whether it is an error or warning is only
                # dependent on how stringent the casting rules are.
                #
                # Will handle the same way.
                try:
                    x /= a
                    assert_equal(x, y / a)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= a
                    assert_equal(xm, y / a)
                    assert_equal(
                        xm.mask,
                        mask_or(mask_or(m, a.mask), (a == t(0)))
                    )
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)

                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, "Failed on type=%s." % t)
                else:
                    assert_equal(len(sup.log), 0, "Failed on type=%s." % t)

    def test_inplace_pow_type(self):
        # Test keeping data w/ (inplace) power
        for t in self.othertypes:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                # Test pow on scalar
                x = uncertain_array([1, 2, 3], mask=[0, 0, 1], dtype=t)
                xx = x ** t(2)
                xx_r = uncertain_array([1, 2 ** 2, 3], mask=[0, 0, 1], dtype=t)
                assert_equal(xx.data, xx_r.data)
                assert_equal(xx.mask, xx_r.mask)
                # Test ipow on scalar
                x **= t(2)
                assert_equal(x.data, xx_r.data)
                assert_equal(x.mask, xx_r.mask)

                assert_equal(len(w), 0, "Failed on type=%s." % t)


class TestUncertainArrayMethods(TestCase):
    # Test class for miscellaneous UncertainArrays methods.
    def setUp(self):
        raise SkipTest
        # Base data definition.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = uncertain_array(data=x, mask=m)
        mX = uncertain_array(data=X, mask=m.reshape(X.shape))
        mXX = uncertain_array(data=XX, mask=m.reshape(XX.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = uncertain_array(data=x, mask=m2)
        m2X = uncertain_array(data=X, mask=m2.reshape(X.shape))
        m2XX = uncertain_array(data=XX, mask=m2.reshape(XX.shape))
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    def test_generic_methods(self):
        # Tests some UncertainArray methods.
        a = uncertain_array([1, 3, 2])
        assert_equal(a.any(), a._data.any())
        assert_equal(a.all(), a._data.all())
        assert_equal(a.argmax(), a._data.argmax())
        assert_equal(a.argmin(), a._data.argmin())
        assert_equal(a.choose(0, 1, 2, 3, 4), a._data.choose(0, 1, 2, 3, 4))
        assert_equal(a.conj(), a._data.conj())
        assert_equal(a.conjugate(), a._data.conjugate())

        m = uncertain_array([[1, 2], [3, 4]])
        assert_equal(m.diagonal(), m._data.diagonal())
        assert_equal(a.sum(), a._data.sum())
        assert_equal(a.take([1, 2]), a._data.take([1, 2]))
        assert_equal(m.transpose(), m._data.transpose())

    def test_allclose(self):
        # Tests allclose on arrays
        a = np.random.rand(10)
        b = a + np.random.rand(10) * 1e-8
        self.assertTrue(allclose(a, b))
        # Test allclose w/ infs
        a[0] = np.inf
        self.assertTrue(not allclose(a, b))
        b[0] = np.inf
        self.assertTrue(allclose(a, b))
        # Test allclose w/ uncertain
        a = uncertain_array(a)
        a[-1] = uncertain
        self.assertTrue(allclose(a, b, uncertain_equal=True))
        self.assertTrue(not allclose(a, b, uncertain_equal=False))
        # Test comparison w/ scalar
        a *= 1e-8
        a[0] = 0
        self.assertTrue(allclose(a, 0, uncertain_equal=True))

        # Test that the function works for MIN_INT integer typed arrays
        a = uncertain_array([np.iinfo(np.int_).min], dtype=np.int_)
        self.assertTrue(allclose(a, a))

    def test_allany(self):
        # Checks the any/all methods/functions.
        x = np.array([[0.13, 0.26, 0.90],
                      [0.28, 0.33, 0.63],
                      [0.31, 0.87, 0.70]])
        m = np.array([[True, False, False],
                      [False, False, False],
                      [True, True, False]], dtype=np.bool_)
        mx = uncertain_array(x, mask=m)
        mxbig = (mx > 0.5)
        mxsmall = (mx < 0.5)

        self.assertFalse(mxbig.all())
        self.assertTrue(mxbig.any())
        assert_equal(mxbig.all(0), [False, False, True])
        assert_equal(mxbig.all(1), [False, False, True])
        assert_equal(mxbig.any(0), [False, False, True])
        assert_equal(mxbig.any(1), [True, True, True])

        self.assertFalse(mxsmall.all())
        self.assertTrue(mxsmall.any())
        assert_equal(mxsmall.all(0), [True, True, False])
        assert_equal(mxsmall.all(1), [False, False, False])
        assert_equal(mxsmall.any(0), [True, True, False])
        assert_equal(mxsmall.any(1), [True, True, False])

    def test_allany_onmatrices(self):
        x = np.array([[0.13, 0.26, 0.90],
                      [0.28, 0.33, 0.63],
                      [0.31, 0.87, 0.70]])
        X = np.matrix(x)
        m = np.array([[True, False, False],
                      [False, False, False],
                      [True, True, False]], dtype=np.bool_)
        mX = uncertain_array(X, mask=m)
        mXbig = (mX > 0.5)
        mXsmall = (mX < 0.5)

        self.assertFalse(mXbig.all())
        self.assertTrue(mXbig.any())
        assert_equal(mXbig.all(0), np.matrix([False, False, True]))
        assert_equal(mXbig.all(1), np.matrix([False, False, True]).T)
        assert_equal(mXbig.any(0), np.matrix([False, False, True]))
        assert_equal(mXbig.any(1), np.matrix([True, True, True]).T)

        self.assertFalse(mXsmall.all())
        self.assertTrue(mXsmall.any())
        assert_equal(mXsmall.all(0), np.matrix([True, True, False]))
        assert_equal(mXsmall.all(1), np.matrix([False, False, False]).T)
        assert_equal(mXsmall.any(0), np.matrix([True, True, False]))
        assert_equal(mXsmall.any(1), np.matrix([True, True, False]).T)

    def test_allany_oddities(self):
        # Some fun with all and any
        store = empty((), dtype=bool)
        full = uncertain_array([1, 2, 3], mask=True)

        self.assertTrue(full.all() is uncertain)
        full.all(out=store)
        self.assertTrue(store)
        self.assertTrue(store._mask, True)
        self.assertTrue(store is not uncertain)

        store = empty((), dtype=bool)
        self.assertTrue(full.any() is uncertain)
        full.any(out=store)
        self.assertTrue(not store)
        self.assertTrue(store._mask, True)
        self.assertTrue(store is not uncertain)

    def test_argmax_argmin(self):
        # Tests argmin & argmax on UncertainArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d

        assert_equal(mx.argmin(), 35)
        assert_equal(mX.argmin(), 35)
        assert_equal(m2x.argmin(), 4)
        assert_equal(m2X.argmin(), 4)
        assert_equal(mx.argmax(), 28)
        assert_equal(mX.argmax(), 28)
        assert_equal(m2x.argmax(), 31)
        assert_equal(m2X.argmax(), 31)

        assert_equal(mX.argmin(0), [2, 2, 2, 5, 0, 5])
        assert_equal(m2X.argmin(0), [2, 2, 4, 5, 0, 4])
        assert_equal(mX.argmax(0), [0, 5, 0, 5, 4, 0])
        assert_equal(m2X.argmax(0), [5, 5, 0, 5, 1, 0])

        assert_equal(mX.argmin(1), [4, 1, 0, 0, 5, 5, ])
        assert_equal(m2X.argmin(1), [4, 4, 0, 0, 5, 3])
        assert_equal(mX.argmax(1), [2, 4, 1, 1, 4, 1])
        assert_equal(m2X.argmax(1), [2, 4, 1, 1, 1, 1])

    def test_clip(self):
        # Tests clip on UncertainArrays.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        m = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        mx = uncertain_array(x, mask=m)
        clipped = mx.clip(2, 8)
        assert_equal(clipped.mask, mx.mask)
        assert_equal(clipped._data, x.clip(2, 8))
        assert_equal(clipped._data, mx._data.clip(2, 8))

    def test_empty(self):
        # Tests empty/like
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        a = uncertain_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')],
                         dtype=datatype)

        b = empty_like(a)
        assert_equal(b.shape, a.shape)

        b = empty(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)

        # check empty_like mask handling
        a = uncertain_array([1, 2, 3], mask=[False, True, False])
        b = empty_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        b = a.view(uncertain_array)
        assert_(np.may_share_memory(a.mask, b.mask))

    def test_put(self):
        # Tests put.
        d = np.arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        x = uncertain_array(d, mask=m)
        self.assertTrue(x[3] is uncertain)
        self.assertTrue(x[4] is uncertain)
        x[[1, 4]] = [10, 40]
        self.assertTrue(x[3] is uncertain)
        self.assertTrue(x[4] is not uncertain)
        assert_equal(x, [0, 10, 2, -1, 40])

        x = uncertain_array(np.arange(10), mask=[1, 0, 0, 0, 0] * 2)
        i = [0, 2, 4, 6]
        x.put(i, [6, 4, 2, 0])
        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9, ]))
        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        x.put(i, uncertain_array([0, 2, 4, 6], [1, 0, 1, 0]))
        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

        x = uncertain_array(np.arange(10), mask=[1, 0, 0, 0, 0] * 2)
        put(x, i, [6, 4, 2, 0])
        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9, ]))
        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        put(x, i, uncertain_array([0, 2, 4, 6], [1, 0, 1, 0]))
        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

    def test_put_nomask(self):
        # GitHub issue 6425
        x = zeros(10)
        z = uncertain_array([3., -1.], mask=[False, True])

        x.put([1, 2], z)
        self.assertTrue(x[0] is not uncertain)
        assert_equal(x[0], 0)
        self.assertTrue(x[1] is not uncertain)
        assert_equal(x[1], 3)
        self.assertTrue(x[2] is uncertain)
        self.assertTrue(x[3] is not uncertain)
        assert_equal(x[3], 0)

    def test_putmask(self):
        x = np.arange(6) + 1
        mx = uncertain_array(x, mask=[0, 0, 0, 1, 1, 1])
        mask = [0, 0, 1, 0, 0, 1]
        # w/o mask, w/o uncertain values
        xx = x.copy()
        putmask(xx, mask, 99)
        assert_equal(xx, [1, 2, 99, 4, 5, 99])
        # w/ mask, w/o uncertain values
        mxx = mx.copy()
        putmask(mxx, mask, 99)
        assert_equal(mxx._data, [1, 2, 99, 4, 5, 99])
        assert_equal(mxx._mask, [0, 0, 0, 1, 1, 0])
        # w/o mask, w/ uncertain values
        values = uncertain_array([10, 20, 30, 40, 50, 60], mask=[1, 1, 1, 0, 0, 0])
        xx = x.copy()
        putmask(xx, mask, values)
        assert_equal(xx._data, [1, 2, 30, 4, 5, 60])
        assert_equal(xx._mask, [0, 0, 1, 0, 0, 0])
        # w/ mask, w/ uncertain values
        mxx = mx.copy()
        putmask(mxx, mask, values)
        assert_equal(mxx._data, [1, 2, 30, 4, 5, 60])
        assert_equal(mxx._mask, [0, 0, 1, 1, 1, 0])
        # w/ mask, w/ uncertain values + hardmask
        mxx = mx.copy()
        putmask(mxx, mask, values)
        assert_equal(mxx, [1, 2, 30, 4, 5, 60])

    def test_ravel(self):
        # Tests ravel
        a = uncertain_array([[1, 2, 3, 4, 5]], mask=[[0, 1, 0, 0, 0]])
        aravel = a.ravel()
        assert_equal(aravel._mask.shape, aravel.shape)
        a = uncertain_array([0, 0], mask=[1, 1])
        aravel = a.ravel()
        assert_equal(aravel._mask.shape, a.shape)
        a = uncertain_array(np.matrix([1, 2, 3, 4, 5]), mask=[[0, 1, 0, 0, 0]])
        aravel = a.ravel()
        assert_equal(aravel.shape, (1, 5))
        assert_equal(aravel._mask.shape, a.shape)
        # Checks that small_mask is preserved
        a = uncertain_array([1, 2, 3, 4], mask=[0, 0, 0, 0], shrink=False)
        assert_equal(a.ravel()._mask, [0, 0, 0, 0])
        a.shape = (2, 2)
        ar = a.ravel()
        assert_equal(ar._mask, [0, 0, 0, 0])
        assert_equal(ar._data, [1, 2, 3, 4])
        # Test index ordering
        assert_equal(a.ravel(order='C'), [1, 2, 3, 4])
        assert_equal(a.ravel(order='F'), [1, 3, 2, 4])

    def test_reshape(self):
        # Tests reshape
        x = np.arange(4)
        x[0] = uncertain
        y = x.reshape(2, 2)
        assert_equal(y.shape, (2, 2,))
        assert_equal(y._mask.shape, (2, 2,))
        assert_equal(x.shape, (4,))
        assert_equal(x._mask.shape, (4,))

    def test_sort(self):
        # Test sort
        x = uncertain_array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)

        sortedx = sort(x)
        assert_equal(sortedx._data, [1, 2, 3, 4])
        assert_equal(sortedx._mask, [0, 0, 0, 1])

        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [4, 1, 2, 3])
        assert_equal(sortedx._mask, [1, 0, 0, 0])

        x.sort()
        assert_equal(x._data, [1, 2, 3, 4])
        assert_equal(x._mask, [0, 0, 0, 1])

        x = uncertain_array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)
        x.sort(endwith=False)
        assert_equal(x._data, [4, 1, 2, 3])
        assert_equal(x._mask, [1, 0, 0, 0])

        x = [1, 4, 2, 3]
        sortedx = sort(x)
        self.assertTrue(not isinstance(sorted, UncertainArray))

        x = uncertain_array([0, 1, -1, -2, 2], mask=nomask, dtype=np.int8)
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [-2, -1, 0, 1, 2])
        x = uncertain_array([0, 1, -1, -2, 2], mask=[0, 1, 0, 0, 1], dtype=np.int8)
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [1, 2, -2, -1, 0])
        assert_equal(sortedx._mask, [1, 1, 0, 0, 0])

    def test_sort_2d(self):
        # Check sort of 2D array.
        # 2D array w/o mask
        a = uncertain_array([[8, 4, 1], [2, 0, 9]])
        a.sort(0)
        assert_equal(a, [[2, 0, 1], [8, 4, 9]])
        a = uncertain_array([[8, 4, 1], [2, 0, 9]])
        a.sort(1)
        assert_equal(a, [[1, 4, 8], [0, 2, 9]])
        # 2D array w/mask
        a = uncertain_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(0)
        assert_equal(a, [[2, 0, 1], [8, 4, 9]])
        assert_equal(a._mask, [[0, 0, 0], [1, 0, 1]])
        a = uncertain_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(1)
        assert_equal(a, [[1, 4, 8], [0, 2, 9]])
        assert_equal(a._mask, [[0, 0, 1], [0, 0, 1]])
        # 3D
        a = uncertain_array([[[7, 8, 9], [4, 5, 6], [1, 2, 3]],
                          [[1, 2, 3], [7, 8, 9], [4, 5, 6]],
                          [[7, 8, 9], [1, 2, 3], [4, 5, 6]],
                          [[4, 5, 6], [1, 2, 3], [7, 8, 9]]])
        a[a % 4 == 0] = uncertain
        am = a.copy()
        an = a.filled(99)
        am.sort(0)
        an.sort(0)
        assert_equal(am, an)
        am = a.copy()
        an = a.filled(99)
        am.sort(1)
        an.sort(1)
        assert_equal(am, an)
        am = a.copy()
        an = a.filled(99)
        am.sort(2)
        an.sort(2)
        assert_equal(am, an)

    def test_sort_flexible(self):
        # Test sort on flexible dtype.
        a = uncertain_array(
            data=[(3, 3), (3, 2), (2, 2), (2, 1), (1, 0), (1, 1), (1, 2)],
            mask=[(0, 0), (0, 1), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0)],
            dtype=[('A', int), ('B', int)])

        test = sort(a)
        b = uncertain_array(
            data=[(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 2), (1, 0)],
            mask=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 0)],
            dtype=[('A', int), ('B', int)])
        assert_equal(test, b)
        assert_equal(test.mask, b.mask)

        test = sort(a, endwith=False)
        b = uncertain_array(
            data=[(1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), ],
            mask=[(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0), ],
            dtype=[('A', int), ('B', int)])
        assert_equal(test, b)
        assert_equal(test.mask, b.mask)

    def test_argsort(self):
        # Test argsort
        a = uncertain_array([1, 5, 2, 4, 3], mask=[1, 0, 0, 1, 0])
        assert_equal(np.argsort(a), argsort(a))

    def test_squeeze(self):
        # Check squeeze
        data = uncertain_array([[1, 2, 3]])
        assert_equal(data.squeeze(), [1, 2, 3])
        data = uncertain_array([[1, 2, 3]], mask=[[1, 1, 1]])
        assert_equal(data.squeeze(), [1, 2, 3])
        assert_equal(data.squeeze()._mask, [1, 1, 1])
        data = uncertain_array([[1]], mask=True)
        self.assertTrue(data.squeeze() is uncertain)

    def test_swapaxes(self):
        # Tests swapaxes on UncertainArrays.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        m = np.array([0, 1, 0, 1, 0, 0,
                      1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0])
        mX = uncertain_array(x, mask=m).reshape(6, 6)
        mXX = mX.reshape(3, 2, 2, 3)

        mXswapped = mX.swapaxes(0, 1)
        assert_equal(mXswapped[-1], mX[:, -1])

        mXXswapped = mXX.swapaxes(0, 2)
        assert_equal(mXXswapped.shape, (2, 2, 3, 3))

    def test_take(self):
        # Tests take
        x = uncertain_array([10, 20, 30, 40], [0, 1, 0, 1])
        assert_equal(x.take([0, 0, 3]), uncertain_array([10, 10, 40], [0, 0, 1]))
        assert_equal(x.take([0, 0, 3]), x[[0, 0, 3]])
        assert_equal(x.take([[0, 1], [0, 1]]),
                     uncertain_array([[10, 20], [10, 20]], [[0, 1], [0, 1]]))

        # assert_equal crashes when passed np.ma.mask
        self.assertIs(x[1], np.ma.uncertain)
        self.assertIs(x.take(1), np.ma.uncertain)

        x = uncertain_array([[10, 20, 30], [40, 50, 60]], mask=[[0, 0, 1], [1, 0, 0, ]])
        assert_equal(x.take([0, 2], axis=1),
                     array([[10, 30], [40, 60]], mask=[[0, 1], [1, 0]]))
        assert_equal(take(x, [0, 2], axis=1),
                     array([[10, 30], [40, 60]], mask=[[0, 1], [1, 0]]))

    def test_take_uncertain_indices(self):
        # Test take w/ uncertain indices
        a = np.array((40, 18, 37, 9, 22))
        indices = np.arange(3)[None,:] + np.arange(5)[:, None]
        mindices = uncertain_array(indices, mask=(indices >= len(a)))
        # No mask
        test = take(a, mindices, mode='clip')
        ctrl = uncertain_array([[40, 18, 37],
                      [18, 37, 9],
                      [37, 9, 22],
                      [9, 22, 22],
                      [22, 22, 22]])
        assert_equal(test, ctrl)
        # Uncertain indices
        test = take(a, mindices)
        ctrl = uncertain_array([[40, 18, 37],
                      [18, 37, 9],
                      [37, 9, 22],
                      [9, 22, 40],
                      [22, 40, 40]])
        ctrl[3, 2] = ctrl[4, 1] = ctrl[4, 2] = uncertain
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        # Uncertain input + uncertain indices
        a = uncertain_array((40, 18, 37, 9, 22), mask=(0, 1, 0, 0, 0))
        test = take(a, mindices)
        ctrl[0, 1] = ctrl[1, 0] = uncertain
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)

    def test_tolist(self):
        # Tests to list
        # ... on 1D
        x = uncertain_array(np.arange(12))
        x[[1, -2]] = uncertain
        xlist = x.tolist()
        self.assertTrue(xlist[1] is None)
        self.assertTrue(xlist[-2] is None)
        # ... on 2D
        x.shape = (3, 4)
        xlist = x.tolist()
        ctrl = [[0, None, 2, 3], [4, 5, 6, 7], [8, 9, None, 11]]
        assert_equal(xlist[0], [0, None, 2, 3])
        assert_equal(xlist[1], [4, 5, 6, 7])
        assert_equal(xlist[2], [8, 9, None, 11])
        assert_equal(xlist, ctrl)
        # ... on structured array w/ uncertain records
        x = uncertain_array(list(zip([1, 2, 3],
                           [1.1, 2.2, 3.3],
                           ['one', 'two', 'thr'])),
                  dtype=[('a', int), ('b', float), ('c', '|S8')])
        x[-1] = uncertain
        assert_equal(x.tolist(),
                     [(1, 1.1, asbytes('one')),
                      (2, 2.2, asbytes('two')),
                      (None, None, None)])
        # ... on structured array w/ uncertain fields
        a = uncertain_array([(1, 2,), (3, 4)], mask=[(0, 1), (0, 0)],
                  dtype=[('a', int), ('b', int)])
        test = a.tolist()
        assert_equal(test, [[1, None], [3, 4]])
        # ... on mvoid
        a = a[0]
        test = a.tolist()
        assert_equal(test, [1, None])

    def test_tolist_specialcase(self):
        # Test mvoid.tolist: make sure we return a standard Python object
        a = uncertain_array([(0, 1), (2, 3)], dtype=[('a', int), ('b', int)])
        # w/o mask: each entry is a np.void whose elements are standard Python
        for entry in a:
            for item in entry.tolist():
                assert_(not isinstance(item, np.generic))
        # w/ mask: each entry is a ma.void whose elements should be
        # standard Python
        a.mask[0] = (0, 1)
        for entry in a:
            for item in entry.tolist():
                assert_(not isinstance(item, np.generic))

    def test_toflex(self):
        # Test the conversion to records
        data = np.arange(10)
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)

        data[[0, 1, 2, -1]] = uncertain
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)

        ndtype = [('i', int), ('s', '|S3'), ('f', float)]
        data = uncertain_array([(i, s, f) for (i, s, f) in zip(np.arange(10),
                                                     'ABCDEFGHIJKLM',
                                                     np.random.rand(10))],
                     dtype=ndtype)
        data[[0, 1, 2, -1]] = uncertain
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)

        ndtype = np.dtype("int, (2,3)float, float")
        data = uncertain_array([(i, f, ff) for (i, f, ff) in zip(np.arange(10),
                                                       np.random.rand(10),
                                                       np.random.rand(10))],
                     dtype=ndtype)
        data[[0, 1, 2, -1]] = uncertain
        record = data.toflex()
        assert_equal_records(record['_data'], data._data)
        assert_equal_records(record['_mask'], data._mask)

    def test_fromflex(self):
        # Test the reconstruction of a uncertain_array from a record
        a = uncertain_array([1, 2, 3])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.mask, a.mask)

        a = uncertain_array([1, 2, 3], mask=[0, 0, 1])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.mask, a.mask)

        a = uncertain_array([(1, 1.), (2, 2.), (3, 3.)], mask=[(1, 0), (0, 0), (0, 1)],
                  dtype=[('A', int), ('B', float)])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.data, a.data)

    def test_arraymethod(self):
        # Test a _arraymethod w/ n argument
        marray = uncertain_array([[1, 2, 3, 4, 5]], mask=[0, 0, 1, 0, 0])
        control = uncertain_array([[1], [2], [3], [4], [5]],
                               mask=[0, 0, 1, 0, 0])
        assert_equal(marray.T, control)
        assert_equal(marray.transpose(), control)

        assert_equal(UncertainArray.cumsum(marray.T, 0), control.cumsum(0))


class TestUncertainArrayMathMethods(TestCase):

    def setUp(self):
        raise SkipTest
        # Base data definition.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = uncertain_array(data=x, mask=m)
        mX = uncertain_array(data=X, mask=m.reshape(X.shape))
        mXX = uncertain_array(data=XX, mask=m.reshape(XX.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = uncertain_array(data=x, mask=m2)
        m2X = uncertain_array(data=X, mask=m2.reshape(X.shape))
        m2XX = uncertain_array(data=XX, mask=m2.reshape(XX.shape))
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    def test_cumsumprod(self):
        # Tests cumsum & cumprod on UncertainArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        mXcp = mX.cumsum(0)
        assert_equal(mXcp._data, mX.filled(0).cumsum(0))
        mXcp = mX.cumsum(1)
        assert_equal(mXcp._data, mX.filled(0).cumsum(1))

        mXcp = mX.cumprod(0)
        assert_equal(mXcp._data, mX.filled(1).cumprod(0))
        mXcp = mX.cumprod(1)
        assert_equal(mXcp._data, mX.filled(1).cumprod(1))

    def test_cumsumprod_with_output(self):
        # Tests cumsum/cumprod w/ output
        xm = uncertain_array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = uncertain

        for funcname in ('cumsum', 'cumprod'):
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)

            # A ndarray as explicit input
            output = np.empty((3, 4), dtype=float)
            output.fill(-9999)
            result = npfunc(xm, axis=0, out=output)
            # ... the result should be the given output
            self.assertTrue(result is output)
            assert_equal(result, xmmeth(axis=0, out=output))

            output = empty((3, 4), dtype=int)
            result = xmmeth(axis=0, out=output)
            self.assertTrue(result is output)

    def test_ptp(self):
        # Tests ptp on UncertainArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        (n, m) = X.shape
        rows = np.zeros(n, np.float)
        cols = np.zeros(m, np.float)

    def test_add_object(self):
        x = uncertain_array(['a', 'b'], mask=[1, 0], dtype=object)
        y = x + 'x'
        assert_equal(y[1], 'bx')
        assert_(y.mask[0])

    def test_sum_object(self):
        # Test sum on object dtype
        a = uncertain_array([1, 2, 3], mask=[1, 0, 0], dtype=np.object)
        assert_equal(a.sum(), 5)
        a = uncertain_array([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.sum(axis=0), [5, 7, 9])

    def test_prod_object(self):
        # Test prod on object dtype
        a = uncertain_array([1, 2, 3], mask=[1, 0, 0], dtype=np.object)
        assert_equal(a.prod(), 2 * 3)
        a = uncertain_array([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.prod(axis=0), [4, 10, 18])

    def test_meananom_object(self):
        # Test mean/anom on object dtype
        a = uncertain_array([1, 2, 3], dtype=np.object)
        assert_equal(a.mean(), 2)
        assert_equal(a.anom(), [-1, 0, 1])

    def test_trace(self):
        # Tests trace on UncertainArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        mXdiag = mX.diagonal()
        assert_almost_equal(mX.trace(),
                            X.trace() - sum(mXdiag.mask * X.diagonal(),
                                            axis=0))
        assert_equal(np.trace(mX), mX.trace())

    def test_dot(self):
        # Tests dot on UncertainArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        fx = mx.filled(0)
        r = mx.dot(mx)
        assert_almost_equal(r.filled(0), fx.dot(fx))
        assert_(r.mask is nomask)

        fX = mX.filled(0)
        r = mX.dot(mX)
        assert_almost_equal(r.filled(0), fX.dot(fX))
        assert_(r.mask[1,3])
        r1 = empty_like(r)
        mX.dot(mX, out=r1)
        assert_almost_equal(r, r1)

        mYY = mXX.swapaxes(-1, -2)
        fXX, fYY = mXX.filled(0), mYY.filled(0)
        r = mXX.dot(mYY)
        assert_almost_equal(r.filled(0), fXX.dot(fYY))
        r1 = empty_like(r)
        mXX.dot(mYY, out=r1)
        assert_almost_equal(r, r1)

    def test_dot_shape_mismatch(self):
        # regression test
        x = uncertain_array([[1,2],[3,4]], mask=[[0,1],[0,0]])
        y = uncertain_array([[1,2],[3,4]], mask=[[0,1],[0,0]])
        z = uncertain_array([[0,1],[3,3]])
        x.dot(y, out=z)
        assert_almost_equal(z.filled(0), [[1, 0], [15, 16]])
        assert_almost_equal(z.mask, [[0, 1], [0, 0]])

    def test_varstd(self):
        # Tests var & std on UncertainArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        assert_equal(mXX.var(axis=3).shape, XX.var(axis=3).shape)
        assert_equal(mX.var().shape, X.var().shape)
        (mXvar0, mXvar1) = (mX.var(axis=0), mX.var(axis=1))

    def test_varstd_specialcases(self):
        # Test a special case for var
        nout = np.array(-1, dtype=float)
        mout = uncertain_array(-1, dtype=float)

        x = uncertain_array(np.arange(10), mask=True)
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            self.assertTrue(method() is uncertain)
            self.assertTrue(method(0) is uncertain)
            self.assertTrue(method(-1) is uncertain)
            # Using a uncertain array as explicit output
            method(out=mout)
            self.assertTrue(mout is not uncertain)
            assert_equal(mout.mask, True)
            # Using a ndarray as explicit output
            method(out=nout)
            self.assertTrue(np.isnan(nout))

        x = uncertain_array(np.arange(10), mask=True)
        x[-1] = 9
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            self.assertTrue(method(ddof=1) is uncertain)
            self.assertTrue(method(0, ddof=1) is uncertain)
            self.assertTrue(method(-1, ddof=1) is uncertain)
            # Using a uncertain array as explicit output
            method(out=mout, ddof=1)
            self.assertTrue(mout is not uncertain)
            assert_equal(mout.mask, True)
            # Using a ndarray as explicit output
            method(out=nout, ddof=1)
            self.assertTrue(np.isnan(nout))

    def test_varstd_ddof(self):
        a = uncertain_array([[1, 1, 0], [1, 1, 0]], mask=[[0, 0, 1], [0, 0, 1]])
        test = a.std(axis=0, ddof=0)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        test = a.std(axis=0, ddof=1)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        test = a.std(axis=0, ddof=2)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [1, 1, 1])

    def test_diag(self):
        # Test diag
        x = np.arange(9).reshape((3, 3))
        x[1, 1] = uncertain
        out = np.diag(x)
        assert_equal(out, [0, 4, 8])
        out = np.diag(out)
        control = uncertain_array([[0, 0, 0], [0, 4, 0], [0, 0, 8]],
                        mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(out, control)

    def test_axis_methods_nomask(self):
        # Test the combination nomask & methods w/ axis
        a = uncertain_array([[1, 2, 3], [4, 5, 6]])

        assert_equal(a.sum(0), [5, 7, 9])
        assert_equal(a.sum(-1), [6, 15])
        assert_equal(a.sum(1), [6, 15])

        assert_equal(a.prod(0), [4, 10, 18])
        assert_equal(a.prod(-1), [6, 120])
        assert_equal(a.prod(1), [6, 120])

        assert_equal(a.min(0), [1, 2, 3])
        assert_equal(a.min(-1), [1, 4])
        assert_equal(a.min(1), [1, 4])

        assert_equal(a.max(0), [4, 5, 6])
        assert_equal(a.max(-1), [3, 6])
        assert_equal(a.max(1), [3, 6])


class TestUncertainArrayMathMethodsComplex(TestCase):
    # Test class for miscellaneous UncertainArrays methods.
    def setUp(self):
        raise SkipTest
        # Base data definition.
        x = np.array([8.375j, 7.545j, 8.828j, 8.5j, 1.757j, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479j,
                      7.189j, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993j])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = uncertain_array(data=x, mask=m)
        mX = uncertain_array(data=X, mask=m.reshape(X.shape))
        mXX = uncertain_array(data=XX, mask=m.reshape(XX.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = uncertain_array(data=x, mask=m2)
        m2X = uncertain_array(data=X, mask=m2.reshape(X.shape))
        m2XX = uncertain_array(data=XX, mask=m2.reshape(XX.shape))
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    def test_varstd(self):
        # Tests var & std on UncertainArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        assert_equal(mXX.var(axis=3).shape, XX.var(axis=3).shape)
        assert_equal(mX.var().shape, X.var().shape)
        (mXvar0, mXvar1) = (mX.var(axis=0), mX.var(axis=1))


class TestUncertainArrayFunctions(TestCase):
    # Test class for miscellaneous functions.

    def setUp(self):
        raise SkipTest
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = uncertain_array(x, mask=m1)
        ym = uncertain_array(y, mask=m2)
        self.info = (xm, ym)

    def test_uncertain_where_bool(self):
        x = [1, 2]
        y = uncertain_where(False, x)
        assert_equal(y, [1, 2])
        assert_equal(y[1], 2)

    def test_uncertain_equal_wlist(self):
        x = [1, 2, 3]
        mx = uncertain_equal(x, 3)
        assert_equal(mx, x)
        assert_equal(mx._mask, [0, 0, 1])
        mx = uncertain_not_equal(x, 3)
        assert_equal(mx, x)
        assert_equal(mx._mask, [1, 1, 0])

    def test_uncertain_where_condition(self):
        # Tests masking functions.
        x = uncertain_array([1., 2., 3., 4., 5.])
        x[2] = uncertain
        assert_equal(uncertain_where(greater(x, 2), x), uncertain_greater(x, 2))
        assert_equal(uncertain_where(greater_equal(x, 2), x),
                     uncertain_greater_equal(x, 2))
        assert_equal(uncertain_where(less(x, 2), x), uncertain_less(x, 2))
        assert_equal(uncertain_where(less_equal(x, 2), x),
                     uncertain_less_equal(x, 2))
        assert_equal(uncertain_where(not_equal(x, 2), x), uncertain_not_equal(x, 2))
        assert_equal(uncertain_where(equal(x, 2), x), uncertain_equal(x, 2))
        assert_equal(uncertain_where(not_equal(x, 2), x), uncertain_not_equal(x, 2))
        assert_equal(uncertain_where([1, 1, 0, 0, 0], [1, 2, 3, 4, 5]),
                     [99, 99, 3, 4, 5])

    def test_uncertain_where_oddities(self):
        # Tests some generic features.
        atest = ones((10, 10, 10), dtype=float)
        btest = zeros(atest.shape, MaskType)
        ctest = uncertain_where(btest, atest)
        assert_equal(atest, ctest)

    def test_uncertain_where_shape_constraint(self):
        a = np.arange(10)
        try:
            test = uncertain_equal(1, a)
        except IndexError:
            pass
        else:
            raise AssertionError("Should have failed...")
        test = uncertain_equal(a, 1)
        assert_equal(test.mask, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_uncertain_otherfunctions(self):
        assert_equal(uncertain_inside(list(range(5)), 1, 3),
                     [0, 199, 199, 199, 4])
        assert_equal(uncertain_outside(list(range(5)), 1, 3), [199, 1, 2, 3, 199])
        assert_equal(uncertain_inside(array(list(range(5)),
                                         mask=[1, 0, 0, 0, 0]), 1, 3).mask,
                     [1, 1, 1, 1, 0])
        assert_equal(uncertain_outside(array(list(range(5)),
                                          mask=[0, 1, 0, 0, 0]), 1, 3).mask,
                     [1, 1, 0, 0, 1])
        assert_equal(uncertain_equal(array(list(range(5)),
                                        mask=[1, 0, 0, 0, 0]), 2).mask,
                     [1, 0, 1, 0, 0])
        assert_equal(uncertain_not_equal(array([2, 2, 1, 2, 1],
                                            mask=[1, 0, 0, 0, 0]), 2).mask,
                     [1, 0, 1, 0, 1])

    def test_round(self):
        a = uncertain_array([1.23456, 2.34567, 3.45678, 4.56789, 5.67890],
                  mask=[0, 1, 0, 0, 0])
        assert_equal(a.round(), [1., 2., 3., 5., 6.])
        assert_equal(a.round(1), [1.2, 2.3, 3.5, 4.6, 5.7])
        assert_equal(a.round(3), [1.235, 2.346, 3.457, 4.568, 5.679])
        b = empty_like(a)
        a.round(out=b)
        assert_equal(b, [1., 2., 3., 5., 6.])

        x = uncertain_array([1., 2., 3., 4., 5.])
        c = uncertain_array([1, 1, 1, 0, 0])
        x[2] = uncertain
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        c[0] = uncertain
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        assert_(z[0] is uncertain)
        assert_(z[1] is not uncertain)
        assert_(z[2] is uncertain)

    def test_round_with_output(self):
        # Testing round with an explicit output

        xm = uncertain_array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = uncertain

        # A ndarray as explicit input
        output = np.empty((3, 4), dtype=float)
        output.fill(-9999)
        result = np.round(xm, decimals=2, out=output)
        # ... the result should be the given output
        self.assertTrue(result is output)
        assert_equal(result, xm.round(decimals=2, out=output))

        output = empty((3, 4), dtype=float)
        result = xm.round(decimals=2, out=output)
        self.assertTrue(result is output)

    def test_round_with_scalar(self):
        # Testing round with scalar/zero dimension input
        # GH issue 2244
        a = uncertain_array(1.1, mask=[False])
        assert_equal(a.round(), 1)

        a = uncertain_array(1.1, mask=[True])
        assert_(a.round() is uncertain)

        a = uncertain_array(1.1, mask=[False])
        output = np.empty(1, dtype=float)
        output.fill(-9999)
        a.round(out=output)
        assert_equal(output, 1)

        a = uncertain_array(1.1, mask=[False])
        output = uncertain_array(-9999., mask=[True])
        a.round(out=output)
        assert_equal(output[()], 1)

        a = uncertain_array(1.1, mask=[True])
        output = uncertain_array(-9999., mask=[False])
        a.round(out=output)
        assert_(output[()] is uncertain)

    def test_identity(self):
        a = identity(5)
        self.assertTrue(isinstance(a, UncertainArray))
        assert_equal(a, np.identity(5))

    def test_power(self):
        x = -1.1
        assert_almost_equal(power(x, 2.), 1.21)
        self.assertTrue(power(x, uncertain) is uncertain)
        x = uncertain_array([-1.1, -1.1, 1.1, 1.1, 0.])
        b = uncertain_array([0.5, 2., 0.5, 2., -1.], mask=[0, 0, 0, 0, 1])
        y = power(x, b)
        assert_almost_equal(y, [0, 1.21, 1.04880884817, 1.21, 0.])
        assert_equal(y._mask, [1, 0, 0, 0, 1])
        b.mask = nomask
        y = power(x, b)
        assert_equal(y._mask, [1, 0, 0, 0, 1])
        z = x ** b
        assert_equal(z._mask, y._mask)
        assert_almost_equal(z, y)
        assert_almost_equal(z._data, y._data)
        x **= b
        assert_equal(x._mask, y._mask)
        assert_almost_equal(x, y)
        assert_almost_equal(x._data, y._data)

    def test_power_with_broadcasting(self):
        # Test power w/ broadcasting
        a2 = np.array([[1., 2., 3.], [4., 5., 6.]])
        a2m = uncertain_array(a2, mask=[[1, 0, 0], [0, 0, 1]])
        b1 = np.array([2, 4, 3])
        b2 = np.array([b1, b1])
        b2m = uncertain_array(b2, mask=[[0, 1, 0], [0, 1, 0]])

        ctrl = uncertain_array([[1 ** 2, 2 ** 4, 3 ** 3], [4 ** 2, 5 ** 4, 6 ** 3]],
                     mask=[[1, 1, 0], [0, 1, 1]])
        # No broadcasting, base & exp w/ mask
        test = a2m ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        # No broadcasting, base w/ mask, exp w/o mask
        test = a2m ** b2
        assert_equal(test, ctrl)
        assert_equal(test.mask, a2m.mask)
        # No broadcasting, base w/o mask, exp w/ mask
        test = a2 ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, b2m.mask)

        ctrl = uncertain_array([[2 ** 2, 4 ** 4, 3 ** 3], [2 ** 2, 4 ** 4, 3 ** 3]],
                     mask=[[0, 1, 0], [0, 1, 0]])
        test = b1 ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        test = b2m ** b1
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)

    def test_where(self):
        # Test the where function
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = uncertain_array(x, mask=m1)
        ym = uncertain_array(y, mask=m2)

        d = where(xm > 2, xm, -9)
        assert_equal(d, [-9., -9., -9., -9., -9., 4.,
                         -9., -9., 10., -9., -9., 3.])
        assert_equal(d._mask, xm._mask)
        d = where(xm > 2, -9, ym)
        assert_equal(d, [5., 0., 3., 2., -1., -9.,
                         -9., -10., -9., 1., 0., -9.])
        assert_equal(d._mask, [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        d = where(xm > 2, xm, uncertain)
        assert_equal(d, [-9., -9., -9., -9., -9., 4.,
                         -9., -9., 10., -9., -9., 3.])
        tmp = xm._mask.copy()
        tmp[(xm <= 2).filled(True)] = True
        assert_equal(d._mask, tmp)

        ixm = xm.astype(int)
        d = where(ixm > 2, ixm, uncertain)
        assert_equal(d, [-9, -9, -9, -9, -9, 4, -9, -9, 10, -9, -9, 3])
        assert_equal(d.dtype, ixm.dtype)

    def test_where_object(self):
        a = np.array(None)
        b = uncertain_array(None)
        r = b.copy()
        assert_equal(np.ma.where(True, a, a), r)
        assert_equal(np.ma.where(True, b, b), r)

    def test_where_with_uncertain_choice(self):
        x = np.arange(10)
        x[3] = uncertain
        c = x >= 8
        # Set False to uncertain
        z = where(c, x, uncertain)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is uncertain)
        assert_(z[4] is uncertain)
        assert_(z[7] is uncertain)
        assert_(z[8] is not uncertain)
        assert_(z[9] is not uncertain)
        assert_equal(x, z)
        # Set True to uncertain
        z = where(c, uncertain, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is uncertain)
        assert_(z[4] is not uncertain)
        assert_(z[7] is not uncertain)
        assert_(z[8] is uncertain)
        assert_(z[9] is uncertain)

    def test_where_with_uncertain_condition(self):
        x = uncertain_array([1., 2., 3., 4., 5.])
        c = uncertain_array([1, 1, 1, 0, 0])
        x[2] = uncertain
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        c[0] = uncertain
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        assert_(z[0] is uncertain)
        assert_(z[1] is not uncertain)
        assert_(z[2] is uncertain)

        x = np.arange(1, 6)
        x[-1] = uncertain
        y = np.arange(1, 6) * 10
        y[2] = uncertain
        c = uncertain_array([1, 1, 1, 0, 0], mask=[1, 0, 0, 0, 0])
        cm = c.filled(1)
        z = where(c, x, y)
        zm = where(cm, x, y)
        assert_equal(z, zm)
        assert_(getmask(zm) is nomask)
        assert_equal(zm, [1, 2, 3, 40, 50])
        z = where(c, uncertain, 1)
        assert_equal(z, [99, 99, 99, 1, 1])
        z = where(c, 1, uncertain)
        assert_equal(z, [99, 1, 1, 99, 99])

    def test_where_type(self):
        # Test the type conservation with where
        x = np.arange(4, dtype=np.int32)
        y = np.arange(4, dtype=np.float32) * 2.2
        test = where(x > 1.5, y, x).dtype
        control = np.find_common_type([np.int32, np.float32], [])
        assert_equal(test, control)

    def test_choose(self):
        # Test choose
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        chosen = choose([2, 3, 1, 0], choices)
        assert_equal(chosen, array([20, 31, 12, 3]))
        chosen = choose([2, 4, 1, 0], choices, mode='clip')
        assert_equal(chosen, array([20, 31, 12, 3]))
        chosen = choose([2, 4, 1, 0], choices, mode='wrap')
        assert_equal(chosen, array([20, 1, 12, 3]))
        # Check with some uncertain indices
        indices_ = uncertain_array([2, 4, 1, 0], mask=[1, 0, 0, 1])
        chosen = choose(indices_, choices, mode='wrap')
        assert_equal(chosen, array([99, 1, 12, 99]))
        assert_equal(chosen.mask, [1, 0, 0, 1])
        # Check with some uncertain choices
        choices = uncertain_array(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1],
                                       [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        chosen = choose(indices_, choices, mode='wrap')
        assert_equal(chosen, array([20, 31, 12, 3]))
        assert_equal(chosen.mask, [1, 0, 0, 1])

    def test_choose_with_out(self):
        # Test choose with an explicit out keyword
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        store = empty(4, dtype=int)
        chosen = choose([2, 3, 1, 0], choices, out=store)
        assert_equal(store, array([20, 31, 12, 3]))
        self.assertTrue(store is chosen)
        # Check with some uncertain indices + out
        store = empty(4, dtype=int)
        indices_ = uncertain_array([2, 3, 1, 0], mask=[1, 0, 0, 1])
        chosen = choose(indices_, choices, mode='wrap', out=store)
        assert_equal(store, array([99, 31, 12, 99]))
        assert_equal(store.mask, [1, 0, 0, 1])
        # Check with some uncertain choices + out ina ndarray !
        choices = uncertain_array(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1],
                                       [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        store = empty(4, dtype=int).view(ndarray)
        chosen = choose(indices_, choices, mode='wrap', out=store)
        assert_equal(store, array([999999, 31, 12, 999999]))

    def test_reshape(self):
        a = np.arange(10)
        a[0] = uncertain
        # Try the default
        b = a.reshape((5, 2))
        assert_equal(b.shape, (5, 2))
        self.assertTrue(b.flags['C'])
        # Try w/ arguments as list instead of tuple
        b = a.reshape(5, 2)
        assert_equal(b.shape, (5, 2))
        self.assertTrue(b.flags['C'])
        # Try w/ order
        b = a.reshape((5, 2), order='F')
        assert_equal(b.shape, (5, 2))
        self.assertTrue(b.flags['F'])
        # Try w/ order
        b = a.reshape(5, 2, order='F')
        assert_equal(b.shape, (5, 2))
        self.assertTrue(b.flags['F'])

        c = np.reshape(a, (2, 5))
        self.assertTrue(isinstance(c, UncertainArray))
        assert_equal(c.shape, (2, 5))
        self.assertTrue(c[0, 0] is uncertain)
        self.assertTrue(c.flags['C'])

    def test_on_ndarray(self):
        # Test functions on ndarrays
        a = np.array([1, 2, 3, 4])
        m = uncertain_array(a, mask=False)
        test = anom(a)
        assert_equal(test, m.anom())
        test = reshape(a, (2, 2))
        assert_equal(test, m.reshape(2, 2))


class TestUncertainFields(TestCase):

    def setUp(self):
        raise SkipTest
        ilist = [1, 2, 3, 4, 5]
        flist = [1.1, 2.2, 3.3, 4.4, 5.5]
        slist = ['one', 'two', 'three', 'four', 'five']
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        mdtype = [('a', bool), ('b', bool), ('c', bool)]
        mask = [0, 1, 0, 0, 1]
        base = uncertain_array(list(zip(ilist, flist, slist)), mask=mask, dtype=ddtype)
        self.data = dict(base=base, mask=mask, ddtype=ddtype, mdtype=mdtype)

    def test_set_records_masks(self):
        base = self.data['base']
        mdtype = self.data['mdtype']
        # Set w/ nomask or uncertain
        base.mask = nomask
        assert_equal_records(base._mask, np.zeros(base.shape, dtype=mdtype))
        base.mask = uncertain
        assert_equal_records(base._mask, np.ones(base.shape, dtype=mdtype))
        # Set w/ simple boolean
        base.mask = False
        assert_equal_records(base._mask, np.zeros(base.shape, dtype=mdtype))
        base.mask = True
        assert_equal_records(base._mask, np.ones(base.shape, dtype=mdtype))
        # Set w/ list
        base.mask = [0, 0, 0, 1, 1]
        assert_equal_records(base._mask,
                             np.array([(x, x, x) for x in [0, 0, 0, 1, 1]],
                                      dtype=mdtype))

    def test_set_record_element(self):
        # Check setting an element of a record)
        base = self.data['base']
        (base_a, base_b, base_c) = (base['a'], base['b'], base['c'])
        base[0] = (pi, pi, 'pi')

        assert_equal(base_a.dtype, int)
        assert_equal(base_a._data, [3, 2, 3, 4, 5])

        assert_equal(base_b.dtype, float)
        assert_equal(base_b._data, [pi, 2.2, 3.3, 4.4, 5.5])

        assert_equal(base_c.dtype, '|S8')
        assert_equal(base_c._data,
                     asbytes_nested(['pi', 'two', 'three', 'four', 'five']))

    def test_set_record_slice(self):
        base = self.data['base']
        (base_a, base_b, base_c) = (base['a'], base['b'], base['c'])
        base[:3] = (pi, pi, 'pi')

        assert_equal(base_a.dtype, int)
        assert_equal(base_a._data, [3, 3, 3, 4, 5])

        assert_equal(base_b.dtype, float)
        assert_equal(base_b._data, [pi, pi, pi, 4.4, 5.5])

        assert_equal(base_c.dtype, '|S8')
        assert_equal(base_c._data,
                     asbytes_nested(['pi', 'pi', 'pi', 'four', 'five']))

    def test_mask_element(self):
        "Check record access"
        base = self.data['base']
        base[0] = uncertain

        for n in ('a', 'b', 'c'):
            assert_equal(base[n].mask, [1, 1, 0, 0, 1])
            assert_equal(base[n]._data, base._data[n])

    def test_getmaskarray(self):
        # Test getmaskarray on flexible dtype
        ndtype = [('a', int), ('b', float)]
        test = empty(3, dtype=ndtype)
        assert_equal(getmaskarray(test),
                     np.array([(0, 0), (0, 0), (0, 0)],
                              dtype=[('a', '|b1'), ('b', '|b1')]))
        test[:] = uncertain
        assert_equal(getmaskarray(test),
                     np.array([(1, 1), (1, 1), (1, 1)],
                              dtype=[('a', '|b1'), ('b', '|b1')]))

    def test_view(self):
        # Test view w/ flexible dtype
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        data = np.array(iterator)
        a = uncertain_array(iterator, dtype=[('a', float), ('b', float)])
        a.mask[0] = (1, 0)
        controlmask = np.array([1] + 19 * [0], dtype=bool)
        # Transform globally to simple dtype
        test = a.view(float)
        assert_equal(test, data.ravel())
        assert_equal(test.mask, controlmask)
        # Transform globally to dty
        test = a.view((float, 2))
        assert_equal(test, data)
        assert_equal(test.mask, controlmask.reshape(-1, 2))

        test = a.view((float, 2), np.matrix)
        assert_equal(test, data)
        self.assertTrue(isinstance(test, np.matrix))

    def test_getitem(self):
        ndtype = [('a', float), ('b', float)]
        a = uncertain_array(list(zip(np.random.rand(10), np.arange(10))), dtype=ndtype)
        a.mask = np.array(list(zip([0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1, 0])),
                          dtype=[('a', bool), ('b', bool)])
        # No mask
        self.assertTrue(isinstance(a[1], UncertainArray))
        # One element uncertain
        self.assertTrue(isinstance(a[0], UncertainArray))
        assert_equal_records(a[0]._data, a._data[0])
        assert_equal_records(a[0]._mask, a._mask[0])
        # All element uncertain
        self.assertTrue(isinstance(a[-2], UncertainArray))
        assert_equal_records(a[-2]._data, a._data[-2])
        assert_equal_records(a[-2]._mask, a._mask[-2])

    def test_setitem(self):
        # Issue 4866: check that one can set individual items in [record][col]
        # and [col][record] order
        ndtype = np.dtype([('a', float), ('b', int)])
        ma = np.ma.UncertainArray([(1.0, 1), (2.0, 2)], dtype=ndtype)
        ma['a'][1] = 3.0
        assert_equal(ma['a'], np.array([1.0, 3.0]))
        ma[1]['a'] = 4.0
        assert_equal(ma['a'], np.array([1.0, 4.0]))
        # Issue 2403
        mdtype = np.dtype([('a', bool), ('b', bool)])
        # soft mask
        control = np.array([(False, True), (True, True)], dtype=mdtype)
        a = np.ma.uncertain_all((2,), dtype=ndtype)
        a['a'][0] = 2
        assert_equal(a.mask, control)
        a = np.ma.uncertain_all((2,), dtype=ndtype)
        a[0]['a'] = 2
        assert_equal(a.mask, control)
        # hard mask
        control = np.array([(True, True), (True, True)], dtype=mdtype)
        a = np.ma.uncertain_all((2,), dtype=ndtype)
        a['a'][0] = 2
        assert_equal(a.mask, control)
        a = np.ma.uncertain_all((2,), dtype=ndtype)
        a[0]['a'] = 2
        assert_equal(a.mask, control)

    def test_element_len(self):
        # check that len() works for mvoid (Github issue #576)
        for rec in self.data['base']:
            assert_equal(len(rec), len(self.data['ddtype']))


class TestUncertainView(TestCase):

    def setUp(self):
        raise SkipTest
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        data = np.array(iterator)
        a = uncertain_array(iterator, dtype=[('a', float), ('b', float)])
        a.mask[0] = (1, 0)
        controlmask = np.array([1] + 19 * [0], dtype=bool)
        self.data = (data, a, controlmask)

    def test_view_to_nothing(self):
        (data, a, controlmask) = self.data
        test = a.view()
        self.assertTrue(isinstance(test, UncertainArray))
        assert_equal(test._data, a._data)
        assert_equal(test._mask, a._mask)

    def test_view_to_type(self):
        (data, a, controlmask) = self.data
        test = a.view(np.ndarray)
        self.assertTrue(not isinstance(test, UncertainArray))
        assert_equal(test, a._data)
        assert_equal_records(test, data.view(a.dtype).squeeze())

    def test_view_to_simple_dtype(self):
        (data, a, controlmask) = self.data
        # View globally
        test = a.view(float)
        self.assertTrue(isinstance(test, UncertainArray))
        assert_equal(test, data.ravel())
        assert_equal(test.mask, controlmask)

    def test_view_to_flexible_dtype(self):
        (data, a, controlmask) = self.data

        test = a.view([('A', float), ('B', float)])
        assert_equal(test.mask.dtype.names, ('A', 'B'))
        assert_equal(test['A'], a['a'])
        assert_equal(test['B'], a['b'])

        test = a[0].view([('A', float), ('B', float)])
        self.assertTrue(isinstance(test, UncertainArray))
        assert_equal(test.mask.dtype.names, ('A', 'B'))
        assert_equal(test['A'], a['a'][0])
        assert_equal(test['B'], a['b'][0])

        test = a[-1].view([('A', float), ('B', float)])
        self.assertTrue(isinstance(test, UncertainArray))
        assert_equal(test.dtype.names, ('A', 'B'))
        assert_equal(test['A'], a['a'][-1])
        assert_equal(test['B'], a['b'][-1])

    def test_view_to_subdtype(self):
        (data, a, controlmask) = self.data
        # View globally
        test = a.view((float, 2))
        self.assertTrue(isinstance(test, UncertainArray))
        assert_equal(test, data)
        assert_equal(test.mask, controlmask.reshape(-1, 2))
        # View on 1 uncertain element
        test = a[0].view((float, 2))
        self.assertTrue(isinstance(test, UncertainArray))
        assert_equal(test, data[0])
        assert_equal(test.mask, (1, 0))
        # View on 1 ununcertain element
        test = a[-1].view((float, 2))
        self.assertTrue(isinstance(test, UncertainArray))
        assert_equal(test, data[-1])

    def test_view_to_dtype_and_type(self):
        (data, a, controlmask) = self.data

        test = a.view((float, 2), np.matrix)
        assert_equal(test, data)
        self.assertTrue(isinstance(test, np.matrix))
        self.assertTrue(not isinstance(test, UncertainArray))

class TestOptionalArgs(TestCase):
    def test_ndarrayfuncs(self):
        # test axis arg behaves the same as ndarray (including mutliple axes)

        d = np.arange(24.0).reshape((2,3,4))
        m = np.zeros(24, dtype=bool).reshape((2,3,4))
        # mask out last element of last dimension
        m[:,:,-1] = True
        a = np.ma.array(d, mask=m)

        def testaxis(f, a, d):
            numpy_f = np.__getattribute__(f)
            ma_f = np.ma.__getattribute__(f)

            # test axis arg
            assert_equal(ma_f(a, axis=1)[...,:-1], numpy_f(d[...,:-1], axis=1))
            assert_equal(ma_f(a, axis=(0,1))[...,:-1],
                         numpy_f(d[...,:-1], axis=(0,1)))

        def testkeepdims(f, a, d):
            numpy_f = np.__getattribute__(f)
            ma_f = np.ma.__getattribute__(f)

            # test keepdims arg
            assert_equal(ma_f(a, keepdims=True).shape,
                         numpy_f(d, keepdims=True).shape)
            assert_equal(ma_f(a, keepdims=False).shape,
                         numpy_f(d, keepdims=False).shape)

            # test both at once
            assert_equal(ma_f(a, axis=1, keepdims=True)[...,:-1],
                         numpy_f(d[...,:-1], axis=1, keepdims=True))
            assert_equal(ma_f(a, axis=(0,1), keepdims=True)[...,:-1],
                         numpy_f(d[...,:-1], axis=(0,1), keepdims=True))

        for f in ['sum', 'prod', 'mean', 'var', 'std']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)

        for f in ['min', 'max']:
            testaxis(f, a, d)

        d = (np.arange(24).reshape((2,3,4))%2 == 0)
        a = np.ma.array(d, mask=m)
        for f in ['all', 'any']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)

    def test_count(self):
        # test np.ma.count specially

        d = np.arange(24.0).reshape((2,3,4))
        m = np.zeros(24, dtype=bool).reshape((2,3,4))
        m[:,0,:] = True
        a = np.ma.array(d, mask=m)

        assert_equal(count(a), 16)
        assert_equal(count(a, axis=1), 2*ones((2,4)))
        assert_equal(count(a, axis=(0,1)), 4*ones((4,)))
        assert_equal(count(a, keepdims=True), 16*ones((1,1,1)))
        assert_equal(count(a, axis=1, keepdims=True), 2*ones((2,1,4)))
        assert_equal(count(a, axis=(0,1), keepdims=True), 4*ones((1,1,4)))
        assert_equal(count(a, axis=-2), 2*ones((2,4)))
        assert_raises(ValueError, count, a, axis=(1,1))
        assert_raises(ValueError, count, a, axis=3)

        # check the 'nomask' path
        a = np.ma.array(d, mask=nomask)

        assert_equal(count(a), 24)
        assert_equal(count(a, axis=1), 3*ones((2,4)))
        assert_equal(count(a, axis=(0,1)), 6*ones((4,)))
        assert_equal(count(a, keepdims=True), 24*ones((1,1,1)))
        assert_equal(np.ndim(count(a, keepdims=True)), 3)
        assert_equal(count(a, axis=1, keepdims=True), 3*ones((2,1,4)))
        assert_equal(count(a, axis=(0,1), keepdims=True), 6*ones((1,1,4)))
        assert_equal(count(a, axis=-2), 3*ones((2,4)))
        assert_raises(ValueError, count, a, axis=(1,1))
        assert_raises(ValueError, count, a, axis=3)

        # check the 'uncertain' singleton
        assert_equal(count(np.ma.uncertain), 0)

        # check 0-d arrays do not allow axis > 0
        assert_raises(ValueError, count, np.ma.array(1), axis=1)


def test_uncertain_array():
    a = np.ma.array([0, 1, 2, 3], mask=[0, 0, 1, 0])
    assert_equal(np.argwhere(a), [[1], [3]])

def test_append_uncertain_array():
    a = np.ma.uncertain_equal([1,2,3], value=2)
    b = np.ma.uncertain_equal([4,3,2], value=2)

    result = np.ma.append(a, b)
    expected_data = [1, 2, 3, 4, 3, 2]
    expected_mask = [False, True, False, False, False, True]
    assert_array_equal(result.data, expected_data)
    assert_array_equal(result.mask, expected_mask)

    a = np.ma.uncertain_all((2,2))
    b = np.ma.ones((3,1))

    result = np.ma.append(a, b)
    expected_data = [1] * 3
    expected_mask = [True] * 4 + [False] * 3
    assert_array_equal(result.data[-3], expected_data)
    assert_array_equal(result.mask, expected_mask)

    result = np.ma.append(a, b, axis=None)
    assert_array_equal(result.data[-3], expected_data)
    assert_array_equal(result.mask, expected_mask)


def test_append_uncertain_array_along_axis():
    a = np.ma.uncertain_equal([1,2,3], value=2)
    b = np.ma.uncertain_values([[4, 5, 6], [7, 8, 9]], 7)

    # When `axis` is specified, `values` must have the correct shape.
    assert_raises(ValueError, np.ma.append, a, b, axis=0)

    result = np.ma.append(a[np.newaxis,:], b, axis=0)
    expected = np.ma.arange(1, 10)
    expected = expected.reshape((3,3))
    assert_array_equal(result.data, expected.data)
    assert_array_equal(result.mask, expected.mask)

###############################################################################
if __name__ == "__main__":
    run_module_suite()
