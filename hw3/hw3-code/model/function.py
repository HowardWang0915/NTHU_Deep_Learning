import numpy as np 

""" 
    This file contains all custom functions to be used
"""

class Conv2d():
    """
        A Conv2d class for out model. The I/O is inspired by TF and Pytorch
    """
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, padding=(0, 0), num_pads=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.num_pads = num_pads
        W, b = self.__init_params()
        self.params = {'W': W, 'b': b}
        self.grads = {'dW': np.zeros(W.shape), 'db': np.zeros(b.shape)}

        # When computing the backward pass, some values are shared with the foward pass
        # Save to improve computaion speed
        # self.cache = None

    def __forward(self, x):
        """
            Forward pass of the conv2d method, 
            :return: output numpy array: Z, shape: (batch_size, #filter, n_H, n_W)
            The logic is inspired by https://hackmd.io/@bouteille/B1Cmns09I,
            a vetorized implementation fo convolution for faster inference speed
        """
        # Get input size
        batch_size, _, H, W = x.shape

        # Compute output height and width
        n_H = int((H - self.kernel_size + 2 * self.num_pads) / self.strides) + 1
        n_W = int((W - self.kernel_size + 2 * self.num_pads) / self.strides) + 1
        
        # import pdb; pdb.set_trace()
        # Initalize output array to zeros
        Z = np.zeros((batch_size, self.out_channels, n_H, n_W))

        # Create x_pad by padding x
        x_pad = self.__pad(x, self.num_pads, self.padding)

        # Convolution steps
        i, j, d = self.__get_indices(x.shape, Z.shape, self.kernel_size, self.strides)
        cols = x_pad[:, d, i, j]
        x_cols = np.concatenate(cols, axis=-1)
        w_col = np.reshape(self.params['W'], (self.out_channels, -1))
        b_col = np.reshape(self.params['b'], (-1, 1))

        # Perform matrix multiplication
        output = np.matmul(w_col, x_cols) + b_col

        # Reshape back matrix to image
        output = np.array(np.hsplit(output, batch_size)).reshape(batch_size, self.out_channels, n_H, n_W)
        assert (output.shape == (batch_size, self.out_channels, n_H, n_W))
        Z = output

        # Save some parameters for computation efficiency
        self.cache = x, x_cols, w_col
        return Z

    def __init_params(self):
        """
            Xavier initialization
        """
        bound = 1 / np.sqrt(self.kernel_size * self.kernel_size)
        w = np.random.uniform(-bound, bound, 
                                    size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        b = np.random.uniform(-bound, bound, size=self.out_channels)
        return w, b

    def __pad(self, x, n, padding):
        """
            Pad zeros to the image, to preserve the output shape.   
        """
        x_pad = np.pad(x, ((0, 0), (0, 0), (n, n), (n, n)), 'constant', constant_values=padding)
        return x_pad

    def __get_indices(self, input_shape, output_shape, kernel_size, stride):
        """
            Compute the index matrices to transform input image into a matrix.
            Since using for loops is to slow, we sacrifice memory for computation.
        """
        # get input size
        m, n_C, n_H, n_W = input_shape

        # get output size
        out_h = output_shape[2]
        out_w = output_shape[3]

        # ----Compute matrix of index i----

        # Level 1 vector.
        level1 = np.repeat(np.arange(kernel_size), kernel_size)
        # Duplicate for the other channels.
        level1 = np.tile(level1, n_C)
        # Create a vector with an increase by 1 at each level.
        everyLevels = stride * np.repeat(np.arange(out_h), out_w)
        # Create matrix of index i at every levels for each channel.
        i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

        # ----Compute matrix of index j----

        # Slide 1 vector.
        slide1 = np.tile(np.arange(kernel_size), kernel_size)
        # Duplicate for the other channels.
        slide1 = np.tile(slide1, n_C)
        # Create a vector with an increase by 1 at each slide.
        everySlides = stride * np.tile(np.arange(out_w), out_h)
        # Create matrix of index j at every slides for each channel.
        j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

        # ----Compute matrix of index d----

        # This is to mark delimitation for each channel
        # during multi-dimensional arrays indexing.
        d = np.repeat(np.arange(n_C), kernel_size * kernel_size).reshape(-1, 1)

        return i, j, d

    def __backward(self, dz):
        """
            Implement backward propagation for a convolutional layer.
        """
        # Load calculated values from cache
        x, x_cols, w_col = self.cache
        output_shape = dz.shape

        # Initialize dx
        dx = np.zeros(x.shape)

        # Pad dx
        dx_pad = self.__pad(dx, self.num_pads, self.padding)

        # Get batch size
        batch_size =  x.shape[0]

        # Compute bias gradient
        self.grads['db'] = np.sum(dz, axis=(0, 2, 3))

        # Reshape dz properly
        dz = np.reshape(dz, (dz.shape[0] * dz.shape[1], dz.shape[2] * dz.shape[3]))
        dz = np.array(np.vsplit(dz, batch_size))
        dz = np.concatenate(dz, axis=-1)

        # Perform matrix multiplication between reshaped dz and w_col to get dx_cols
        dx_cols = np.matmul(w_col.T, dz)
        # Weight gradient
        dw_col = np.matmul(dz, x_cols.T)
        
        i, j, d = self.__get_indices(x.shape, output_shape, self.kernel_size, self.strides)
        dx_cols_reshaped = np.array(np.hsplit(dx_cols, batch_size))
        # Reshape matrix back to image
        np.add.at(dx_pad, (slice(None), d, i, j), dx_cols_reshaped)
        # Remove padding from new image
        dx = dx_pad[:, :, self.num_pads:-self.num_pads, self.num_pads:-self.num_pads]

        # Reshape dw_col into dw
        self.grads['dW'] = np.reshape(dw_col, (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

        # Final check the output shape is correct
        assert(dx.shape == x.shape)

        return dx
        
    def forward(self, x):
        return self.__forward(x)

    def backward(self, x):
        return self.__backward(x)

class ReLU():
    """
        ReLU activation function
    """
    def __init__(self):
        pass
        # self.cache = None

    def __forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def __backward(self, dz):
        x = self.cache
        dx = dz * np.where(x <= 0, 0, 1)
        return dx

    def forward(self, x):
        return self.__forward(x)

    def backward(self, dz):
        return self.__backward(dz)

def softmax(x):
    """
        Compute softmax of x
    """
    exps = np.exp(x)
    sum_exps = np.reshape(np.sum(exps, axis=1), (-1, 1))
    # return exps / np.sum(exps, axis=0)
    return exps / sum_exps

class Dense():
    """
        Fully connected layer
    """
    def __init__(self, in_dims, out_dims):
        self.in_dims = in_dims
        self.out_dims = out_dims
        W, b = self.__init_params()
        self.params = {"W": W, "b": b}
        self.grads = {'dW': np.zeros(self.params['W'].shape),
                      'db': np.zeros(self.params['b'].shape)}
        # self.cache = None 

    def __init_params(self):
        W = np.random.randn(self.out_dims, self.in_dims) * np.sqrt(1. / self.in_dims)
        b = np.zeros((1, self.out_dims))
        return W, b

    def __forward(self, x):
        """
            Forward pass for fully connected layer.
        """
        self.cache = x
        z = np.matmul(x, self.params['W'].T) + self.params['b']
        return z

    def __backward(self, dz):
        x = self.cache
        batch_size = x.shape[0]

        self.grads['dW'] = (1. / batch_size) * np.matmul(dz.T, x)
        self.grads['db'] = (1. / batch_size) * np.sum(dz, axis=0, keepdims=True)

        dx = np.matmul(dz, self.params['W'])
        return dx

    def forward(self, x):
        return self.__forward(x)

    def backward(self, dz):
        return self.__backward(dz)

class Flatten():
    """
        Because we are dealing with images, we must 
        reshape the 3d features into a single column vector
    """
    def __init__(self):
        pass

    def __forward(self, x):
        self.forward_shape = x.shape
        x_flatten = np.reshape(x, (self.forward_shape[0], -1))
        return x_flatten

    def __backward(self, dz):
        dz = np.reshape(dz, self.forward_shape)
        return dz

    def forward(self, x):
        return self.__forward(x)

    def backward(self, dz):
        return self.__backward(dz)
