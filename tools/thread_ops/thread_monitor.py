# thread_monitor.py
def is_alive(thread):
    return thread.is_alive()

def get_status(thread):
    return "Alive" if thread.is_alive() else "Dead"