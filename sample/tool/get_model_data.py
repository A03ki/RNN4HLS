class GetModelData:
    """
    モデルの各パラメーターについての情報を保持する。

    Attributes
    ----------
    model : torch.nn.Module
        PyTorchで作ったモデル。
    modules : list of str
        model内に含まれる各層。
    name_params: list of str
        各層のパラメーターの名前。
    params : list of torch
        各層のパラメータ。
    size_shape : list of tuple
        各層の転置済みパラメーターのサイズ。
    size_flatten: list of int
        各層の転置済みパラメーターを一次元配列にした時の長さ。
    """
    def __init__(self, model):
        self.model = model
        self.modules = self.get_module_list()
        self.name_params, self.params = self.get_param_list()
        self.size_shape, self.size_flatten = self.get_size_list()

    def get_param_list(self):
        name_params = []
        params = []
        param_name_gene = self.model.named_parameters()
        for module in self.modules:
            if module.startswith('lstm'):
                for name, param in param_name_gene:
                    name_sp = name.split(".")[2]
                    if 'bias_ih' not in name:
                        name_params.append(f'{module}_{name_sp}')
                        params.append(param)
                    else:
                        name_next, param_next = next(param_name_gene)
                        name_params.append(f'{module}_'
                                           + f'{name_sp.replace("_ih", "")}')
                        params.append(param + param_next)
                        break

            elif module.startswith('linear'):
                for name, param in param_name_gene:
                    name_params.append(f'{module}_{name.split(".")[2]}')
                    params.append(param)
                    if 'bias' in name:
                        break
        return name_params, params

    def get_size_list(self):
        size_shape = []
        size_flatten = []
        for _param in self.params:
            param = _param.detach().numpy().T
            size_shape.append(param.shape)
            size_flatten.append(len(param.flatten()))
        return size_shape, size_flatten

    def param_to_str1d(self, param):
        param_1d = param.detach().numpy().T.flatten()
        param_1d_str_list = ["%.9f" % i for i in param_1d]
        return param_1d_str_list

    def get_module_list(self):
        modules = []
        model_name = self.model.__class__.__name__.lower()
        for name in self.model.named_modules():
            class_name = name[1].__class__.__name__.lower()
            if class_name != 'modulelist' and class_name != model_name:
                for module_name in reversed(modules):
                    if module_name.startswith(class_name):
                        # get last module-number in this class_name
                        next_num = int(module_name.replace(class_name, '')) + 1
                        modules.append(class_name + str(next_num))
                        break
                else:
                    modules.append(class_name + '0')
        return modules


class GetDefinition(GetModelData):

    def __init__(self, model, input_x_row):
        super().__init__(model)
        self.size_output = []
        self.variable_size_str = self.get_definition_variable_size(input_x_row)
        self.macro_str = self.get_definition_macro_flatten_len(self.size_output)
        self.variable_params_str = self.get_definition_variable_params()

    def get_definition_macro_flatten_len(self, size_output):
        macro_str = ''
        assert size_output != [], 'size_output is empty list'
        for name, size in zip(self.name_params, self.size_flatten):
            macro_str += f'#define {name.upper()} {size}\n'
        macro_str += f'#define INPUT_X {self.size_input}\n'
        for module, size in zip(self.modules, size_output):
            macro_str += f'#define OUTPUT_{module.upper()} {size}\n'
        macro_str += f'#define INPUT_COLUMN {self.size_shape[1][0]}\n'

        return macro_str

    def get_definition_variable_size(self, input_x_row):
        name_iter = iter(self.name_params)
        shape_iter = iter(self.size_shape)
        size_str = ''
        size_def = ['int {0[0]}_row = {1};\n',
                    'int {0[0]}_matrix_k = {1};\n',
                    'int {0[0]}_column = {1};\n']
        self.size_output = []
        for name, shape in zip(name_iter, shape_iter):
            if name.startswith('lstm') and 'weight_ih' in name:
                if 'lstm0_' not in name:
                    input_x_row = int(self.size_output[-1] / shape[0])
                name_sp = name.split('_')
                size_str += size_def[0].format(name_sp, input_x_row)
                size_str += size_def[1].format(name_sp, shape[0])
                shape_next = next(shape_iter)
                next(name_iter)
                size_str += size_def[2].format(name_sp, shape_next[0])
                self.size_output.append(input_x_row * shape_next[0])
            elif name.startswith('linear') and 'weight' in name:
                if 'linear0_' not in name:
                    self.size_output[-1] = input_x_row * shape[1]
                input_x_row = int(self.size_output[-1] / shape[0])
                size_str += size_def[0].format(name.split('_'), input_x_row)
                size_str += size_def[1].format(name.split('_'), shape[0])
                size_str += size_def[2].format(name.split('_'), shape[1])
                self.size_output.append(input_x_row * shape[1])
        return size_str

    def get_definition_variable_params(self):
        params_str = ''
        for name, param in zip(self.name_params, self.params):
            params_str += f'float {name}[{name.upper()}]' \
                + ' = {' + ','.join(self.param_to_str1d(param)) + '};\n'
        return params_str
