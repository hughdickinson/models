#!/usr/bin/env python
"""Construct an RGB raster image from three-band CCD images

RGBImage class is used to generate an RGB raster image from a set of
three images corresponding to red, green, and blue bands.  The custom
mapping function can be used to bring out details you need to see in
the final RGB image.

USAGE

Here is an example usage.  Say you want to create a JPEG image from a
set of three HST ACS images stored in FITS files.  The following set
of instructions will load 2D images into (NumPy) array objects,
corresponding to red, green, and blue bands:

  import pyfits as P
  hdus1 = P.open('f814.fits')
  hdus2 = P.open('f606.fits')
  hdus3 = P.open('f450.fits')
  img1,img2,img3 = hdus1[0].data,hdus2[0].data,hdus3[0].data

Now feed these images into RGBImage object:

  from RGBImage import *
  im = RGBImage(img1,img2,img3,
                scales=[7000,4000,15000],
                mapping=map_Lupton04,
                beta=2.)

In the above, the counts in each band image are scaled by given
factors.  The mapping is given by map_Lupton04 function (which
implements Lupton et al. (2004) algorithm), with a customized
nonlinearity factor (beta=2.).

To previw the resulting RGB image, do:

  im.show()

To save the resulting image as a JPEG file with the highest quality,
do:

  im.save_as('test.jpg',quality=100)

Note that a custom mapping function can be supplied by users, as long
as the following prototype is follows:

  def map_NAME(r,g,b,args={}):
    (... do mapping ...)
    return r,g,b

The argument args is a dictionary which should contain the variable
parameters used in the mapping, if necessary.  (Detail: RGBImage class
sends all the optinal arguments supplied at the construction to the
mapping function as a dictionary.)

REQUIREMENTS

  The following Python modules:
    -- Image (Python Imaging Library; tested with v1.1.4)
    -- pyfits (PyFITS; tested with v1.1b2)
    -- numpy (NumPy; tested with v0.9.9.2614)

REFERENCE

Lupton et al. (2004), PASP, 116, 133

The implementation closely follows the IDL procedure DJS_RGB_MAKE by
David Schlegel, which is build upon NW_RGB_MAKE by Nick Wherry.

TODO

More mapping functions need to be implemented (currently only Lupton
et al. (2004) and square root mappings have been coded up).

Handling of saturated pixels has not been implemented for Lupton04
mapping.

Work on better documentation.

HISTORY

  June 22, 2006 (ver 0.1) -- Implements the most essential part of the
  class.
"""
__version__ = '0.1 (June 22, 2006)'
__credits__ = '''The code is written by Taro Sato (nomo17k@gmail.com)'''

from astropy.io import fits as P
import tensorflow as tf


def map_Lupton04(imagesTensor, beta=3., alpha=0.06, Q=3.5,
                 bandScalings=[1.000,1.176,1.818],args={}):
    """Lupton et al. (2004) mapping

    DESCRIPTION

      First define radius: rad = beta * (r + g + b).  Then the mapping is
      given by

        R = r*f(rad), G = g*f(rad), B = b*f(rad) ,

      where

               / 0,                     x <= 0 ,
        f(x) = |
               \ arcsinh(x) / x,        x > 0 .

      Hence the mapped values are NOT normalized to [0,1].  Saturation and
      max(R,G,B) > 1 cases have not been taken care of.
    """
    # r,g,b = N.asarray(r).copy(),N.asarray(g).copy(),N.asarray(b).copy()
    # make sure all the entries are >= 0
    imagesTensor[:, :, :, 0:3] /= 255.0
    imagesTensor[:, :, :, 0:3] *= tf.convert_to_tensor(bandScalings)
    imagesTensor[:, :, :, 0:3] = tf.where(images[:, :, :, 0:3] > 0.0, images[:, :, :, 0:3], 0.0)
    # r = N.where(r>0., r, 0.)
    # g = N.where(g>0., g, 0.)
    # b = N.where(b>0., b, 0.)
    # compute nonlinear mapping
    radius = tf.reduce_sum(images[:, :, :, 0:3], axis=3)/beta
    nlfac = tf.where(radius > 0.0, tf.math.asinh(alpha*Q*radius)/(Q*radius), 0.0)
    # radius_ok = radius > 0.0
    # nlfac = radius * 0.0
    # nlfac[radius_ok] = N.arcsinh(radius[radius_ok])/radius[radius_ok]
    imagesTensor[:, :, :, 0:3] *= nlfac
    # r = r*nlfac
    # g = g*nlfac
    # b = b*nlfac
    # if args.get('desaturate'):
    #     # optionally desaturate pixels that are dominated by a single
    #     # colour to avoid colourful speckled sky
    #     a = (r+g+b)/3.0
    #     N.putmask(a, a == 0.0, 1.0)
    #     rab = r / a / beta
    #     gab = g / a / beta
    #     bab = b / a / beta
    #     mask = N.array((rab, gab, bab))
    #     w = N.max(mask, 0)
    #     N.putmask(w, w > 1.0, 1.0)
    #     w = 1 - w
    #     w = N.sin(w*N.pi/2.0)
    #     r = r*w + a*(1-w)
    #     g = g*w + a*(1-w)
    #     b = b*w + a*(1-w)
    # # optionally add a grey pedestal
    # if args.has_key('pedestal'):
    #     pedestal = args['pedestal']
    #     r += pedestal
    #     g += pedestal
    #     b += pedestal
    return imagesTensor * 255.0
