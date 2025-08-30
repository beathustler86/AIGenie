# thread_registry.py
registry = {}

def register_thread(name, thread):
    registry[name] = thread
    print(f"[REGISTRY] Registered: {name}")

def get_thread(name):
    return registry.get(name)

def list_threads():
    return list(registry.keys())

def thread_status(name):
    thread = registry.get(name)
    if thread:
        return "Alive" if thread.is_alive() else "Dead"
    return "Not Found"