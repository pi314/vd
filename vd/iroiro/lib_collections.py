from .internal_utils import exporter
export, __all__ = exporter()


@export
class namablelist(list):
    def __init__(self, *args, **kwargs):
        super().__setattr__('_name_to_index', {})
        super().__setattr__('_index_to_name', {})

        def indexof(name):
            if isinstance(name, int):
                return None if name >= len(self) else name
            return self._name_to_index.get(name)
        indexof.__dict__ = self._name_to_index
        super().__setattr__('indexof', indexof)

        if args and kwargs:
            raise ValueError('Cannot mix named and unnamed values')

        if args:
            super().__init__(args[0])

        if kwargs:
            for idx, (key, value) in enumerate(kwargs.items()):
                self.append(value)
                self.nameit(idx, key)

    def nameof(self, index):
        if isinstance(index, str):
            if index in self._name_to_index:
                return index
            else:
                return None
        try:
            return self._index_to_name[index]
        except KeyError:
            return None

    def nameit(self, index, name):
        self._name_to_index[name] = index
        self._index_to_name[index] = name

    def unname(self, name):
        index = self.indexof(name)
        del self._name_to_index[name]
        del self._index_to_name[index]

    def _norm_idx(self, index, _keyerror):
        if isinstance(index, int):
            ret = index
        elif isinstance(index, str):
            ret = self.indexof(index)
            if ret is None:
                raise _keyerror(index)
        elif isinstance(index, slice):
            ret = slice(self._norm_idx(index.start, _keyerror),
                        self._norm_idx(index.stop, _keyerror),
                        index.step)
        else:
            raise TypeError('index should be in int, str, or slice')
        return ret

    def _getitem(self, index, _keyerror):
        return super().__getitem__(self._norm_idx(index, _keyerror))

    def _setitem(self, index, value, _keyerror):
        return super().__setitem__(self._norm_idx(index, _keyerror), value)

    def __getitem__(self, index):
        return self._getitem(index, _keyerror=KeyError)

    def __setitem__(self, index, value):
        return self._setitem(index, value, _keyerror=KeyError)

    def __getattr__(self, attr):
        return self._getitem(attr, _keyerror=AttributeError)

    def __setattr__(self, attr, value):
        return self._setitem(attr, value, _keyerror=AttributeError)

    def __dir__(self):
        return dir(super()) + list(self._name_to_index.keys())

    def keys(self):
        return list(self._name_to_index.keys())

    def values(self):
        return list(self)
