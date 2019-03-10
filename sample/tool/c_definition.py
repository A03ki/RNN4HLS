from tool.model_data import LayerName, Parameters

from itertools import product
from more_itertools import chunked
from numpy import float32, ndim
from pprint import pformat


def define_value(dtype, front, rear):
    return f'{dtype} {front} = {rear};\n'


def define_macro(front, rear):
    return f'#define {front} {rear}\n'


def change_param_str(array):
    dim = ndim(array)
    if dim == 1:
        array = list(float32(array))
    elif dim == 2:
        array = [list(i) for i in float32(array)]
    param_str = pformat(array,
                        compact=True).replace('[', '{').replace(']', '}')
    return param_str


class Definition:

    def __init__(self, model, input_x_row):
        self.model = model
        self.input_x_row = input_x_row
        self.modules = LayerName(self.model).get_name_counted()
        parameters = Parameters(self.model)
        self.params = parameters.get_params_1d()
        self.param_names = parameters.get_param_names()
        self.size_flatten = parameters.get_flatten_size()
        self.size_shape = parameters.get_shape_size()

    def get_macro_hedder(self):
        macro_str = ''
        _, size_output = self.comb_matrix_named_size()
        for name, size in zip(self.param_names, self.size_flatten):
            macro_str += define_macro(name.upper(), size)
        macro_str += define_macro('INPUT_X', self.input_x_row)
        for module, size in zip(self.modules, size_output):
            macro_str += define_macro(f'OUTPUT_{module.upper()}', size)
        macro_str += define_macro('OUTPUT_LSTM0_COLUMN', self.size_shape[1][0])

        return macro_str

    def get_variable_size(self):
        size_str = ''
        matrix_sizes, _ = self.comb_matrix_named_size()
        for i, k in matrix_sizes:
            size_str += define_value('int', i, k)

        return size_str

    def comb_matrix_named_size(self):
        matrix_sizes = []
        output_sizes = []
        row_size = self.input_x_row
        names = self.add_rowcolumn(self.modules)
        sizes = self.put_matrix_size(self.param_names, self.size_shape)
        for _names, _sizes in zip(chunked(names, 3), chunked(sizes, 2)):
            row_str, matrix_k_str, column_str = _names
            matrix_k_size, column_size = _sizes
            if output_sizes != []:
                row_size = int(output_sizes[-1] / matrix_k_size)
            matrix_sizes.append((row_str, row_size))
            matrix_sizes.append((matrix_k_str, matrix_k_size))
            matrix_sizes.append((column_str, column_size))
            output_sizes.append((row_size * column_size))

        return matrix_sizes, output_sizes

    def add_rowcolumn(self, modules):
        rowcolumn = ['row', 'matrix_k', 'column']
        modules_rowcolumn = []
        for module, rowcom in product(modules, rowcolumn):
            modules_rowcolumn.append(f'{module}_{rowcom}')

        return modules_rowcolumn

    def put_matrix_size(self, names, shapes):
        matrix_k_column_size = []
        for name, shape in zip(names, shapes):
            if name.startswith('lstm') and 'bias' not in name:
                matrix_k_column_size.append(shape[0])
            elif name.startswith('linear') and 'bias' not in name:
                matrix_k_column_size.append(shape[0])
                matrix_k_column_size.append(shape[1])

        return matrix_k_column_size

    def get_variable_params(self):
        params_str = ''
        for name, param in zip(self.param_names, self.params):
            params_str += define_value('float', f'{name}[{name.upper()}]',
                                       change_param_str(param))

        return params_str
