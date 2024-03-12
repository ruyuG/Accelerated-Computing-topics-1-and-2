import sys
from LebwohlLasher_cython_openMP import main

# Check if the correct number of arguments is provided
if len(sys.argv) == 5:
    # Unpack
    _, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG = sys.argv
    
    # Call the main function with the provided arguments
    main(_, int(ITERATIONS), int(SIZE), float(TEMPERATURE), int(PLOTFLAG))
else:
    # If not, print the correct usage of the script
    print("Usage: {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
