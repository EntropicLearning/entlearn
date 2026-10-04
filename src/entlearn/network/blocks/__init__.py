"""Private fitted blocks with direct lifecycle methods.

Each block owns its complete update and loss contribution, taking the operation's fit
session explicitly. Coordinate helpers remain module functions where they are tested
separately; shared tensor mathematics lives in primitives.
"""
