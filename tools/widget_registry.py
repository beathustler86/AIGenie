_widget_map = {}

def register_widget(name, widget):
    _widget_map[name] = widget

def get_widget(name):
    return _widget_map.get(name)

def register_callback(widget_name, callback):
    widget = get_widget(widget_name)
    if widget:
        widget.clicked.connect(callback)