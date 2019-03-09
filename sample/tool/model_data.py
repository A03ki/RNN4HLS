from collections import Counter


class LayerName:

    def __init__(self, model):
        self.model = model

    def get_name_counted(self):
        modules = self.get_name_list()
        counter = Counter(modules)
        for i, name in enumerate(reversed(modules)):
            counter[name] -= 1
            modules[-i+1] = name + str(counter[name])

        return modules

    def get_name_list(self):
        modules = []
        model_name = self.model.__class__.__name__.lower()
        for name in self.model.named_modules():
            class_name = name[1].__class__.__name__.lower()
            if class_name != 'modulelist' and class_name != model_name:
                modules.append(class_name)

        return modules


class Parameters:

    def __init__(self, model):
        self.model = model
        self.layer_name = LayerName(self.model)

    def get_params_1d(self):
        params = self.get_params()
        return [param.detach().numpy().T.flatten() for param in params]

    def get_params_2d(self):
        params = self.get_params()
        return [param.detach().numpy().T for param in params]

    def get_params(self):
        name_params = self._comb_param_name()
        params = []
        params_gene = self.model.parameters()
        for name, param in zip(name_params, params_gene):
            if 'bias_ih' not in name or not name.startswith('lstm'):
                params.append(param)
            else:  # 2つのバイアスは統合して使う
                param_next = next(params_gene)
                params.append(param + param_next)

        return params

    def get_param_names(self):

        return self._change_bias()

    def _comb_param_name(self):
        _param_names = []
        modules = self.layer_name.get_name_counted()
        names = self.model.named_parameters()
        for module in modules:
            for name, _ in names:
                _param_names.append(f'{module}_{name.split(".")[2]}')
                if 'bias_hh' in name:
                    break
                elif name.endswith('bias'):
                    break

        return _param_names

    def _remove_bias_ih(self):
        _param_names = self._comb_param_name()
        for name in _param_names[:]:
            if 'bias_ih' in name and name.startswith('lstm'):
                _param_names.remove(name)

        return _param_names

    def _change_bias(self):
        _rv_names = self._remove_bias_ih()
        for i, name in enumerate(_rv_names):
            if 'bias_hh' in name and name.startswith('lstm'):
                _rv_names[i] = '{0[0]}_{0[1]}_{0[3]}'.format(name.split('_'))

        return _rv_names

    def get_flatten_size(self):
        flatten_size = []
        for param in self.get_params_1d():
            flatten_size.append(len(param))

        return flatten_size

    def get_shape_size(self):
        shape_size = []
        for param in self.get_params_2d():
            shape_size.append(param.shape)

        return shape_size
