def execfile(filepath, globals=None, locals=None):
    if globals is None:
        globals = {}
    globals.update({
        "__file__": filepath,
        "__name__": "__main__",
    })
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), globals, locals)


print("\nChoose a Model to run\n\n1. Activate Mask Detection (Press q to exit)\n2. Automated Attendance System (Press q to exit)")


inp = int( input("\nEnter your choice "))

def one():
    return execfile("mask/detect_mask_video.py")
 
def two():
    return execfile("face/AttendanceFR.py")


switcher = {
        1: one,
        2: two,
        }



def numbers_to_strings(inp):
    # Get the function from switcher dictionary
    func = switcher.get(inp, "nothing")
    # Execute the function
    return func()

numbers_to_strings(inp)


