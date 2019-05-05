class DimensionError(TypeError):
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        super(DimensionError, self).__init__()

    @property
    def message(self):
        return ("Image has incompatible dimensionality: Expected dimension is {0}D and you provided a {1}D "
                "image.".format(self.obj1, self.obj2))

    def __str__(self):
        return self.message


class ShapeError(TypeError):
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        super(ShapeError, self).__init__()

    @property
    def message(self):
        return ("Image has incompatible shape: Expected shape is {0} and you provided an image with shape {1}"
                .format(tuple(self.obj1), tuple(self.obj2)))

    def __str__(self):
        return self.message


class AffineError(TypeError):
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2

        super(AffineError, self).__init__()

    @property
    def message(self):
        return ("Image has incompatible affine: Expected affine is: \n{0}\n\n and you provided an image with "
                "affine: \n{1}"
                .format(self.obj1, self.obj2))

    def __str__(self):
        return self.message
