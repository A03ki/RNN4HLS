from collections import Counter


class GetLayerName:

    def __init__(self, model):
        self.model = model

    def name_counted(self):
        modules = self._name_list(self.model)
        counter = Counter(modules)
        for i, name in enumerate(reversed(modules)):
            counter[name] -= 1
            modules[-i+1] = name + str(counter[name])
        return modules

    def _name_list(self, model):
        modules = []
        model_name = model.__class__.__name__.lower()
        for name in model.named_modules():
            class_name = name[1].__class__.__name__.lower()
            if class_name != 'modulelist' and class_name != model_name:
                modules.append(class_name)
        return modules
