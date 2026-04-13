__version__ = '0.3.1'

__all__ = []

def check_python_version():
    import sys
    assert sys.version_info.major, sys.version_info.minor >= (3, 7)

check_python_version()
del check_python_version


def load_internal_modules(prefix, post_load_hook):
    import importlib
    import pkgutil
    scope_prefix = __name__ + '.'
    for mod_info in pkgutil.iter_modules(__path__, scope_prefix):
        mod_name = mod_info.name[len(scope_prefix):]
        if not mod_name.startswith(prefix):
            continue

        module = importlib.import_module(mod_info.name)
        post_load_hook(module, mod_name)


def post_load_lib(module, mod_name):
    global __all__

    ext_name = mod_name[4:]

    if '__all__' in module.__dict__:
        attrs = module.__dict__['__all__']
    else:
        attrs = [x for x in module.__dict__ if not x.startswith('_')]

    __all__ += attrs

    # Register module into package namespace with external name
    globals()[ext_name] = module
    globals().update({attr: getattr(module, attr) for attr in attrs})

    # Delete old name from package namespace
    del globals()[mod_name]


def post_load_bin(module, mod_name):
    ext_name = mod_name[4:]

    # Register module into iroiro.bin with external name
    setattr(bin, ext_name, module)

    # Delete old name from package namespace
    del globals()[mod_name]


from . import bin

load_internal_modules('lib_', post_load_lib)
load_internal_modules('bin_', post_load_bin)

del post_load_lib
del post_load_bin
del load_internal_modules
