import json

from process_bigraph import allocate_core, Composite


def run(document=None, time=None):
    if not document:
        return

    if isinstance(document, str):
        with open(document, 'r') as path:
            document = json.load(path)

    core = allocate_core()
    composite = Composite(
        document,
        core=core)

    if not time:
        composite.run(0.0)
    else:
        composite.run(time)
        
    print(composite.read_bridge())

    
if __name__ == '__main__':
    import fire
    fire.Fire(run)
