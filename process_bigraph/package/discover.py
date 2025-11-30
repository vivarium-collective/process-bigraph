import importlib.metadata
import pkgutil
import inspect
from pprint import pprint as pp

from process_bigraph import Process, Step, ProcessTypes


def recursive_dynamic_import(core, package_name: str) -> list[tuple[str, Process | Step ]]:
    classes_to_import = []
    adjusted_package_name: str = package_name.replace("-", "_")

    # TODO: fix module name discovery based on package name
    if adjusted_package_name == "vivarium_interface":
        adjusted_package_name = "vivarium"
    if adjusted_package_name == "sed2_demo":
        adjusted_package_name = "sed2"

    try:
        module = importlib.import_module(adjusted_package_name)
    except ModuleNotFoundError:
        # TODO: Add code to try and find correct module name via accessing `top_level.txt`,
        #  and getting the correct module name
        # find top-level.txt
        # find correct module name
        # return recursive_dynamic_import(correct_module_name)
        raise ModuleNotFoundError(f"Error: module `{adjusted_package_name}` not found when trying to dynamically import!")

    if hasattr(module, 'register_types'):
        core = module.register_types(core)

    class_members = inspect.getmembers(module, inspect.isclass)
    for class_name, cls in class_members:
        if cls == Process:
            continue
        if cls == Step:
            continue
        if issubclass(cls, Process):
            classes_to_import.append((f"{package_name}.{class_name}", cls))
        if issubclass(cls, Step):
            classes_to_import.append((f"{package_name}.{class_name}", cls))

    modules_to_check = pkgutil.iter_modules(module.__path__) if hasattr(module, '__path__') else []

    for _module_loader, subname, isPkg in modules_to_check:
        classes_to_import += recursive_dynamic_import(core, f"{adjusted_package_name}.{subname}")

    return classes_to_import


def import_package_modules(core, name, package):
    import ipdb; ipdb.set_trace()


def is_process_library(package: importlib.metadata.Distribution) -> bool:
    for entry in ([] if package.requires is None else package.requires):
        if "process-bigraph" in entry:
            return True

    return False


def load_local_modules(core) -> list[tuple[str, Process | Step ]]:
    packages = importlib.metadata.distributions()
    processes = []
    for package in packages:
        if not is_process_library(package):
            continue
        processes += recursive_dynamic_import(core, package.name)
 
    return processes


def import_processes(core, name):
    processes = {}
    module = importlib.import_module(name)
    for attr in dir(module):
        entry = getattr(module, attr)
        if inspect.isclass(entry) and issubclass(entry, (Process, Step)):
            processes[attr] = entry
    return processes


def traverse_modules(core) -> list[tuple[str, Process | Step ]]:
    processes = {}
    package_modules = pkgutil.walk_packages(['.'])

    for _, module_name, is_package in package_modules:
        processes.update(import_processes(core, module_name))

    return processes


def discover_packages(core) -> ProcessTypes:
    for name, process in load_local_modules(core):
        if name not in core.process_registry.registry:
            core.register_process(name, process)

    processes = traverse_modules(core)
    core.register_processes(processes)

    return core

