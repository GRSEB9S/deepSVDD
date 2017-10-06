import theano
import lasagne
import numpy as np

from six import integer_types
from six.moves import xrange
import six.moves.builtins as builtins

from theano import Op, tensor, Variable, Apply
from theano.tensor.signal.pool import PoolGrad, Pool

from theano.sandbox.cuda import register_opt
from theano.gof import local_optimizer
from theano.sandbox.cuda.basic_ops import HostFromGpu, gpu_contiguous,\
    host_from_gpu
from theano.sandbox.cuda import dnn_available
from theano.sandbox.cuda.dnn import GpuDnnPoolGrad, dnn_pool


def patch_lasagne():
    """
    patch Lasagne function to create shared parameters
    in order to avoid broadcasting issues
    """

    lasagne.utils.create_param = create_param


def create_param(spec, shape, name=None):
    """
    Helper method to create Theano shared variables for layer parameters
    and to initialize them.
    Parameters
    ----------
    spec : scalar number, numpy array, Theano expression, or callable
        Either of the following:
        * a scalar or a numpy array with the initial parameter values
        * a Theano expression or shared variable representing the parameters
        * a function or callable that takes the desired shape of
          the parameter array as its single argument and returns
          a numpy array, a Theano expression, or a shared variable
          representing the parameters.
    shape : iterable of int
        a tuple or other iterable of integers representing the desired
        shape of the parameter array.
    name : string, optional
        The name to give to the parameter variable. Ignored if `spec`
        is or returns a Theano expression or shared variable that
        already has a name.
    Returns
    -------
    Theano shared variable or Theano expression
        A Theano shared variable or expression representing layer parameters.
        If a scalar or a numpy array was provided, a shared variable is
        initialized to contain this array. If a shared variable or expression
        was provided, it is simply returned. If a callable was provided, it is
        called, and its output is used to initialize a shared variable.
    Notes
    -----
    This function is called by :meth:`Layer.add_param()` in the constructor
    of most :class:`Layer` subclasses. This enables those layers to
    support initialization with scalars, numpy arrays, existing Theano shared
    variables or expressions, and callables for generating initial parameter
    values, Theano expressions, or shared variables.
    """
    import numbers  # to check if argument is a number
    shape = tuple(shape)  # convert to tuple if needed
    if any(d <= 0 for d in shape):
        raise ValueError((
            "Cannot create param with a non-positive shape dimension. "
            "Tried to create param with shape=%r, name=%r") % (shape, name))

    err_prefix = "cannot initialize parameter %s: " % name
    if callable(spec):
        spec = spec(shape)
        err_prefix += "the %s returned by the provided callable"
    else:
        err_prefix += "the provided %s"

    if isinstance(spec, numbers.Number) or isinstance(spec, np.generic) \
            and spec.dtype.kind in 'biufc':
        spec = np.asarray(spec)

    if isinstance(spec, np.ndarray):
        if spec.shape != shape:
            raise ValueError("%s has shape %s, should be %s" %
                             (err_prefix % "numpy array", spec.shape, shape))
        # We assume parameter variables do not change shape after creation.
        # We can thus fix their broadcast pattern, to allow Theano to infer
        # broadcastable dimensions of expressions involving these parameters.

        # change here: remove broadcasting even if dimension is 1
        # bcast = tuple(s == 1 for s in shape)

        bcast = tuple(False for s in shape)
        spec = theano.shared(spec, broadcastable=bcast)

    if isinstance(spec, theano.Variable):
        # We cannot check the shape here, Theano expressions (even shared
        # variables) do not have a fixed compile-time shape. We can check the
        # dimensionality though.
        if spec.ndim != len(shape):
            raise ValueError("%s has %d dimensions, should be %d" %
                             (err_prefix % "Theano variable", spec.ndim,
                              len(shape)))
        # We only assign a name if the user hasn't done so already.
        if not spec.name:
            spec.name = name
        return spec

    else:
        if "callable" in err_prefix:
            raise TypeError("%s is not a numpy array or a Theano expression" %
                            (err_prefix % "value"))
        else:
            raise TypeError("%s is not a numpy array, a Theano expression, "
                            "or a callable" % (err_prefix % "spec"))


def my_pool_2d(input, ds, ignore_border=None, st=None, padding=(0, 0),
               mode='max'):
    """
    This function is a patch to the maxpool op of Theano:
    contrarily to current implementation of maxpool, the gradient is backpropagated
    to only one input of a given patch if several inputs have the same value. This is
    consistent with the CuDNN implementation (and therefore the op is replaced by the
    CuDNN version when possible).
    """

    if input.ndim < 2:
        raise NotImplementedError('pool_2d requires a dimension >= 2')

    if not ignore_border is None:
        # check that ignore_border is True if provided
        assert ignore_border
    ignore_border = True

    if input.ndim == 4:
        op = MyPool(ds, ignore_border, st=st, padding=padding, mode=mode)
        output = op(input)
        return output

    # extract image dimensions
    img_shape = input.shape[-2:]

    # count the number of "leading" dimensions, store as dmatrix
    batch_size = tensor.prod(input.shape[:-2])
    batch_size = tensor.shape_padright(batch_size, 1)

    # store as 4D tensor with shape: (batch_size,1,height,width)
    new_shape = tensor.cast(tensor.join(0, batch_size,
                                        tensor.as_tensor([1]),
                                        img_shape), 'int64')
    input_4D = tensor.reshape(input, new_shape, ndim=4)

    # downsample mini-batch of images
    op = MyPool(ds, ignore_border, st=st, padding=padding, mode=mode)
    output = op(input_4D)

    # restore to original shape
    outshp = tensor.join(0, input.shape[:-2], output.shape[-2:])
    return tensor.reshape(output, outshp, ndim=input.ndim)


class MyPool(Pool):
    """
    Overwrite Pool with MyPool class
    """

    def __init__(self, ds, ignore_border=True, st=None, padding=(0, 0),
                 mode='max'):

        Pool.__init__(self, ds, ignore_border=True, st=None, padding=(0, 0),
                      mode='max')

    def grad(self, inp, grads):
        """
        override grad function to call MyMaxPoolGrad
        """
        x, = inp
        gz, = grads
        assert self.mode == 'max'
        maxout = self(x)
        return [MyMaxPoolGrad(self.ds,
                              ignore_border=self.ignore_border,
                              st=self.st, padding=self.padding)(x, maxout, gz)]


class MyMaxPoolGrad(PoolGrad):
    def __init__(self, ds, ignore_border, st=None, padding=(0, 0)):

        PoolGrad.__init__(self, ds, ignore_border, st, padding, mode='max')

    def make_node(self, x, maxout, gz):
        # make_node should only be called by the grad function of
        # Pool, so these asserts should not fail.
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(maxout, Variable) and maxout.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        x = tensor.as_tensor_variable(x)
        maxout = tensor.as_tensor_variable(maxout)
        gz = tensor.as_tensor_variable(gz)

        return Apply(self, [x, maxout, gz], [x.type()])

    def perform(self, node, inp, out):

        print("Careful I am using python to compute maxpool gradients")

        assert self.mode == 'max'
        x, maxout, gz = inp
        gx_stg, = out
        # number of pooling output rows
        pr = maxout.shape[-2]
        # number of pooling output cols
        pc = maxout.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w

        # pad the image
        if self.padding != (0, 0):
            y = np.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)] = x
        else:
            y = x
        gx = np.zeros_like(y)

        for n in xrange(x.shape[0]):
            for k in xrange(x.shape[1]):
                for r in xrange(pr):
                    row_st = builtins.max(r * st0, self.padding[0])
                    row_end = builtins.min(row_st + ds0, img_rows)
                    for c in xrange(pc):
                        col_st = builtins.max(c * st1, self.padding[1])
                        col_end = builtins.min(col_st + ds1, img_cols)
                        break_flag = False
                        for row_ind in xrange(row_st, row_end):
                            for col_ind in xrange(col_st, col_end):
                                if (maxout[n, k, r, c] == y[n, k, row_ind, col_ind]):
                                    gx[n, k, row_ind, col_ind] += gz[n, k, r, c]
                                    # addition here
                                    break_flag = True
                                    break
                            if break_flag:
                                break
        # unpad the image
        gx = gx[:, :, pad_h:(img_rows - pad_h), pad_w:(img_cols - pad_w)]
        gx_stg[0] = gx

    def grad(self, inp, grads):

        # addition here
        raise NotImplementedError("Second order gradient not implemented for custom MaxPool")

    def c_code(self, node, name, inp, out, sub):
        print("Compiled C code for custom max pooling.")
        assert self.mode == 'max'
        x, z, gz = inp
        gx, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        return """
        // sanity checks
        int x_typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
        int z_typenum = PyArray_ObjectType((PyObject*)%(z)s, 0);
        int gz_typenum = PyArray_ObjectType((PyObject*)%(gz)s, 0);
        if ((x_typenum != z_typenum) || (x_typenum != gz_typenum))
        {
            PyErr_SetString(PyExc_ValueError, "input types must all match");
            %(fail)s;
        }
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(z)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "z must be a 4d ndarray");
            %(fail)s;
        }
        if(PyArray_NDIM(%(gz)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "gz must be a 4d ndarray");
            %(fail)s;
        }
        int z_r, z_c;
        z_r = PyArray_DIMS(%(z)s)[2];
        z_c = PyArray_DIMS(%(z)s)[3];
        int r, c; // shape of the padded_input
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += %(pd0)s * 2;
        c += %(pd1)s * 2;
        // allocating memory for gx
        if ((!%(gx)s)
          || !PyArray_ISCONTIGUOUS(%(gx)s)
          || *PyArray_DIMS(%(gx)s)!=4
          ||(PyArray_DIMS(%(gx)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(gx)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(gx)s)[2] != PyArray_DIMS(%(x)s)[2])
          ||(PyArray_DIMS(%(gx)s)[3] != PyArray_DIMS(%(x)s)[3])
          )
        {
          Py_XDECREF(%(gx)s);
          %(gx)s = (PyArrayObject*) PyArray_ZEROS(4, PyArray_DIMS(%(x)s), x_typenum,0);
        }
        else {
          PyArray_FILLWBYTE(%(gx)s, 0);
        }
        int r_st, r_end, c_st, c_end; // used to index into the input img x
        dtype_%(z)s maximum; // temp var for maximum value in a region
        if (z_r && z_c)
        {
            for(int b=0; b<PyArray_DIMS(%(x)s)[0]; b++){
              for(int k=0; k<PyArray_DIMS(%(x)s)[1]; k++){
                for(int i=0; i< z_r; i++){
                  r_st = i * %(st0)s;
                  r_end = r_st + %(ds0)s;
                  // skip the padding
                  r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
                  r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
                  // from padded_img space to img space
                  r_st -= %(pd0)s;
                  r_end -= %(pd0)s;
                  for(int j=0; j<z_c; j++){
                    c_st = j * %(st1)s;
                    c_end = c_st + %(ds1)s;
                    // skip the padding
                    c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
                    c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
                    // change coordinates from padding_img space into img space
                    c_st -= %(pd1)s;
                    c_end -= %(pd1)s;
                    // the maximum value
                    maximum = ((dtype_%(z)s*)(PyArray_GETPTR4(%(z)s,b,k,i,j)))[0];
                    // the gradient corresponding to this maximum value in z
                    dtype_%(gz)s * gz = (
                          (dtype_%(gz)s*)(PyArray_GETPTR4(%(gz)s, b, k, i, j)));
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        dtype_%(gx)s * gx = (
                          (dtype_%(gx)s*)(PyArray_GETPTR4(%(gx)s, b, k, m, n)));
                        if (a == maximum){
                          gx[0] = gx[0] + gz[0];
                          // addition here
                          m = r_end;
                          n = c_end;
                          // end addition
                        }
                      }
                    }
                  }
                }
              }
            }
        }
        """ % locals()

    def c_code_cache_version(self):
        # NB: changed (0, 7)
        return (0, 7)


""" Replace ops by CuDNN versions when possible
"""


@register_opt('cudnn')
@local_optimizer([MyPool])
def local_mypool_dnn_alternative(node):
    if not dnn_available():
        return
    if isinstance(node.op, MyPool):
        if not node.op.ignore_border:
            return
        img, = node.inputs
        ds = node.op.ds
        stride = node.op.st
        pad = node.op.padding
        mode = node.op.mode
        if (img.owner and isinstance(img.owner.op, HostFromGpu)):
            ret = dnn_pool(gpu_contiguous(img.owner.inputs[0]),
                           ds, stride=stride, pad=pad, mode=mode)
            return [host_from_gpu(ret)]


@register_opt('cudnn')
@local_optimizer([MyMaxPoolGrad])
def local_mypool_dnn_grad_stride(node):
    if not dnn_available():
        return
    if isinstance(node.op, MyMaxPoolGrad):
        if not node.op.ignore_border:
            return
        inp, out, inp_grad = node.inputs
        ds = node.op.ds
        st = node.op.st
        pad = node.op.padding
        mode = node.op.mode

        if ((inp.owner and isinstance(inp.owner.op, HostFromGpu)) or
            (out.owner and isinstance(out.owner.op, HostFromGpu)) or
            (inp_grad.owner and isinstance(inp_grad.owner.op,
                                           HostFromGpu))):

            ret = GpuDnnPoolGrad(mode=mode)(gpu_contiguous(inp),
                                            gpu_contiguous(out),
                                            gpu_contiguous(inp_grad),
                                            ds, st, pad)
            return [host_from_gpu(ret)]
