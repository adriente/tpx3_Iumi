import numpy as np
from numpy.typing import NDArray
from typing import TypeVar
from numba import njit

# Custom types for numba usage.
IA = NDArray[np.uint64]
IB = NDArray[np.bool]
UnSigned = TypeVar("UnSigned", IA, np.uint64)
Bool = TypeVar("Bool", IB, np.bool)

# ToA : Time of Arrival
# ToT : Time over Threshold
# TDC : Time to Digital Converter

@njit
def get_block(v: UnSigned, width: int, shift: int) -> UnSigned:
    """
    Selects a bit sequence from a 64 bits unsigned integer (or 64 bits integer array).

    Parameters
    ----------
    v : UnSigned
        integer input
    width : int
        length of the selected bit sequence 
    shift : int
        starting value of the sequence

    Results
    -------
    seq : np.uint64
        selected bits sequence
    """
    return v >> np.uint64(shift) & np.uint64(2**width - 1)

@njit
def is_tdc(data : UnSigned) -> Bool:
    """
    True -> The data are a TDC
    False -> The data are not a TDC
    """
    blocks = get_block(data, 4,60)
    return blocks == 6

@njit
def get_toa(data : UnSigned) -> UnSigned:
    """
    Returns the ToA value in 25 ns unit.
    """
    blocks = get_block(data, 14,30)
    return blocks

@njit
def get_tdc_type(data : UnSigned) -> UnSigned :
    """
    Returns the type of TDC.
    """
    blocks = get_block(data, 4,56)
    return blocks

@njit
def get_tdc_time(data : UnSigned) -> UnSigned :
    """
    Returns the TDC time value in 3.125 ns unit.
    """
    blocks = get_block(data, 35,9)
    return blocks

@njit
def get_fast_tdc_time(data : UnSigned) -> UnSigned : 
    """
    Returns the fast TDC time value in 260.4166 ps unit.
    """
    blocks = get_block(data, 4,5)
    return blocks

@njit
def get_spidr_time(data : UnSigned) -> UnSigned :
    """
    Return the SPIDR time in 0.4096 ms unit.
    """
    spidr = get_block(data, 16,0)
    return spidr

@njit
def get_ftoa(data : UnSigned) -> UnSigned :
    """
    Return the fast ToA time in negative 1.5625 ns unit
    """
    fine = get_block(data, 4, 16)
    return fine

@njit
def get_tdc_counter(data : UnSigned) -> UnSigned :
    """
    Return the TDC counter value
    """
    blocks = get_block(data, 12,44)
    return blocks

@njit
def get_ci(data : UnSigned) -> UnSigned :
    """
    Return the chip index
    """
    block = get_block(data, 8, 32)
    return block

@njit
def is_header(data : UnSigned) -> Bool :
    """
    True -> the data are header
    False -> the data are not header
    """
    block = get_block(data, 32, 0)
    return block == 861425748

@njit
def is_hit(data : UnSigned) -> Bool :
    """
    True -> the data are pixel hit
    False -> the data are not pixel hit
    """
    block = get_block(data,4,60)
    return block == 11

@njit
def get_hits_number(data : UnSigned) -> int :
    """
    Return the number of pixel hit in the data
    """
    tot = 0
    for i in data :
        if is_hit(i) :
            tot +=1
    return tot

@njit
def get_tot(data : UnSigned) -> UnSigned :
    """
    Return the Tot in 25 ns unit
    """
    block = get_block(data, 10,20)
    return block

@njit
def get_xy(data: UnSigned,ci : int) -> tuple :
    """
    Return the x and y coordinates of the pixel hit

    Parameters
    ----------
    data : Unsigned 
        tpx3 bytes to decode
    ci : int
        Cheetah3 chip index

    Returns
    -------
    coordinates : tuple
        (x,y) coordinates of the pixel hit

    Notes
    -----
    This only works for a specific chip indices configuration. 
    TODO : generalise for any chip index config.
    """
    pi = get_block(data,3,44)
    spi = get_block(data,6,47)
    col = get_block(data,7,53)
    if pi < 4 :
        sub_x = 0
    else :
        sub_x = 1
    if pi < 4 :
        sub_y = pi
    else :
        sub_y = pi-4
    if ci == 3 :
        y = round(4*spi + sub_y)
        x = round(2*col + sub_x)
        cx, cy = x,y
    if ci == 0 :
        y = round(4*spi+ sub_y)
        x = round( 2*col + sub_x)
        cx, cy = x+256, y
    if ci == 2 :
        y = round(255 - 4*spi - sub_y)
        x = round(255 - 2*col - sub_x)
        cx, cy = x, y +256
    if ci == 1 :
        y = round(255 - 4*spi - sub_y)
        x = round(255 - 2*col - sub_x)
        cx, cy = x+256, y+256
    return (cx,cy)

@njit
def read_tpx3_bytes_no_tdc(data: UnSigned) -> tuple :
    """
    Return the time and position values of pixel hit data. No correction is applied.
    The TDC are not taken into account.

    Parameters
    ----------
    data : Unsigned
        tpx3 raw data, 64 bits int or sequence of 64 bits int
    
    Results
    -------
    output : tuple
        (
        hit_times in ns,
        ToT in ns,
        x positions in pixels,
        y positions in pixels
        )
    """
    hit_number = get_hits_number(data)
    times = np.zeros((hit_number,),dtype = np.uint64)
    tot = np.zeros((hit_number,), dtype=np.uint64)
    x = np.zeros((hit_number,),dtype = np.uint64)
    y = np.zeros((hit_number,),dtype = np.uint64)
    max_toa = np.uint64(2**34)
    ro_state = 0
    ro_count = 0
    index = 0
    ci = 0
    for d in data :
        if is_header(d) :
            ci = get_ci(d)
        if is_hit(d) :
            late_hit = 0
            spidr = np.uint64(get_spidr_time(d))
            rough = ((spidr << np.uint(14)) | get_toa(d)) & np.uint(0x3FFFFFFF)
            fine = get_ftoa(d)
            hit_time = (rough << np.uint(4)) - fine
            if 1.0*hit_time > 0.95*max_toa and ro_state==0 :
                ro_state = 1
            if ro_state==1 and 1.0*hit_time < 0.05*max_toa :
                ro_state = 2
                ro_count+=1
            if ro_state ==2 :
                if 1.0*hit_time > 0.95*max_toa :
                    late_hit = 1
                elif 1.0*hit_time > 0.05*max_toa :
                    ro_state = 0
            ro_hit_time = np.uint64(np.round((hit_time + (ro_count-late_hit)*max_toa)*1.5625))
            times[index] = ro_hit_time
            tot[index] = get_tot(d)*25
            x[index], y[index] = get_xy(d,ci)
            index += 1
    return times, tot, x, y

def read_tpx3_no_tdc(file_path : str) -> tuple[np.array] :
    """
    Loads a .tpx3 file and returns time and position values of pixel hit data. No correction is applied.
    The TDC are not taken into account.

    Parameters
    ----------
    file_path : Path
        path of the file to open
    
    Results
    -------
    output : tuple
        (hit times, ToTs, xs, ys)
    """
    with open(file_path, "rb") as f:
        data  = np.frombuffer(f.read(), dtype=np.uint64)
    times, tot, x, y = read_tpx3_bytes_no_tdc(data)
    return times, tot, x, y

def save_tpx3_no_tdc(output_filename : str, input_file_path : str) -> None :
    """
    Reads an input .tpx3 file (see read_tpx3_no_tdc). Saves it as a .npy file.
    output_array[0] : Times of Arrival in ns unit
    ouput_array[1] : Times over Threshold in ns unit
    output_array[2] : x positions in pixels
    output_array[3] : y positions in pixels
    """
    result_tuple = read_tpx3_no_tdc(input_file_path)
    np.save(output_filename,np.array(result_tuple))