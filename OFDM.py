import numpy as np

def OFDM_Transmitter(Data_in_IFFT, FFT_Size, CP_Size): 
    data2CP = np.fft.ifft(Data_in_IFFT, axis=0)
    data2CP = np.sqrt(FFT_Size) * data2CP

    # --- insert CP --- 

    cyclic_prefix = data2CP[-CP_Size:, :] 

    Transmitted_Signal = np.vstack((cyclic_prefix, data2CP)) 

    # --- P2S --- 
    Transmitted_Signal = Transmitted_Signal.reshape(-1, order='F') 

    return Transmitted_Signal, data2CP

def OFDM_Receiver(Received_Signal, FFT_Size, CP_Size, Length_of_symbol): 
    # --- Serial to Parallel ---
    Received_Signal = Received_Signal.reshape((-1, Length_of_symbol), order='F') 

    # --- Remove Cyclic Prefix ---
    Received_signal_removed_CP = Received_Signal[CP_Size:, :] 

    # --- FFT ---
    Unrecovered_Signal = np.fft.fft(Received_signal_removed_CP, axis=0) 
    Unrecovered_Signal *= (1 / np.sqrt(FFT_Size)) 

    return Unrecovered_Signal 
