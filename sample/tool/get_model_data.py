class GetModelData:

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
                    if 'bias_ih' not in name:
                        name_params.append('{0}_{1[2]}'.format(module,
                                           name.split('.')))
                        params.append(param)
                    else:
                        name_next, param_next = next(param_name_gene)
                        name_params.append('{0}_{1[2]}'.format(module,
                                           name.replace('_ih', '').split('.')))
                        params.append(param + param_next)
                        break

            elif module.startswith('linear'):
                for name, param in param_name_gene:
                    name_params.append('{0}_{1[2]}'.format(module,
                                       name.split('.')))
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
        self.macro_len_str = self.get_definition_macro_flatten_len()
        self.variable_size_str = self.get_definition_variable_size(input_x_row)
        self.variable_params_str = self.get_definition_variable_params()

    def get_definition_macro_flatten_len(self):
        name_params_str = ''
        for name, size in zip(self.name_params, self.size_flatten):
            name_params_str += '#define {0} {1}\n'.format(name.upper(), size)
        return name_params_str

    def get_definition_variable_size(self, input_x_row):
        name_iter = iter(self.name_params)
        shape_iter = iter(self.size_shape)
        size_str = ''
        size_def = ['int {0[0]}_row = {1};\n',
                    'int {0[0]}_matrix_k = {1};\n',
                    'int {0[0]}_column = {1};\n']
        size_input = 0
        for name, shape in zip(name_iter, shape_iter):
            if name.startswith('lstm') and 'weight_ih' in name:
                if 'lstm0_' not in name:
                    input_x_row = int(size_input / shape[0])
                size_str += size_def[0].format(name.split('_'), input_x_row)
                size_str += size_def[1].format(name.split('_'), shape[0])
                shape_next = next(shape_iter)
                next(name_iter)
                size_str += size_def[2].format(name.split('_'), shape_next[0])
                size_input = input_x_row * shape_next[0]
            elif name.startswith('linear') and 'weight' in name:
                if 'linear0' not in name:
                    size_input = input_x_row * shape[1]
                input_x_row = int(size_input / shape[0])
                size_str += size_def[0].format(name.split('_'), input_x_row)
                size_str += size_def[1].format(name.split('_'), shape[0])
                size_str += size_def[2].format(name.split('_'), shape[1])
                size_input = shape[0] * shape[1]
        return size_str

    def get_definition_variable_params(self):
        params_str = ''
        for name, param in zip(self.name_params, self.params):
            params_str += 'float {0}[{1}]'.format(name, name.upper()) \
                + ' = {' + ','.join(self.param_to_str1d(param)) + '};\n'
        return params_str
