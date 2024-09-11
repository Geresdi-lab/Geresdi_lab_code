import numbers
import Geresdi_lab_code.DC.dc_setup as DC_setup
import qcodes.instrument_drivers
import zhinst.qcodes.driver
from multimethod import multimethod
import datetime
from time import sleep
import numpy as np
import warnings
from typing import Union, Tuple
import os
'''
Future possible additions:
include maximum sweeping speed
'''

class Station:
    def __init__( self, ivvi = '', k1 = '', k2 = '', lockin = '', lake = '', mag_x = '', mag_y = '', mag_z = '', k3 = '' ):
        self.ivvi = ivvi
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.lockin = lockin
        self.lake = lake
        self.mag_x = mag_x
        self.mag_y = mag_y
        self.mag_z = mag_z
        self.figure = False
        self.fig = ''
        self.pcolor = 'bo'  


    def InsertText(
                    self,
                    text: str,
                    filepath: str ):
        '''
        At the given file path in inserts the text at the beginning of the file
        '''
        with open( filepath, 'r+') as file:
            content = file.read()
            file.seek(0, 0)
            file.write( text + content )

    def open_figure( self ):
               
        self.figure = True
        self.c_idx = 0
        
        plt.close()
        fig = plt.figure()
        fig.clf()
        
        self.fig = fig

#----------------------------------------------------------------------------------------------------------------------------------------        
    def draw_figure(
                    self,
                    x,
                    y,
                    x_l: str,
                    y_l: str,
                    ):
               
        plt.plot( x, y, self.pcolor, markersize = 4 )
        plt.xlabel( x_l )
        plt.ylabel( y_l )
        plt.tight_layout()
        plt.ticklabel_format( scilimits=(-3, 4) )
        self.fig.canvas.draw()

#----------------------------------------------------------------------------------------------------------------------------------------        
                
    def draw_figure_y1y2(
                        self,
                        x,
                        y1,
                        y2,
                        x_l: str,
                        y1_l: str,
                        y2_l: str,
                    ):
               
        
        self.fig.set_size_inches( 8, 3 )
        plt.subplot( 1, 2, 1 )
        plt.xlabel( x_l )
        plt.ylabel( y1_l )
        plt.plot(x, y1, self.pcolor, markersize = 4 )
        plt.tight_layout()
        plt.ticklabel_format( scilimits=(-3, 4) )

        plt.subplot( 1, 2, 2 ) 
        plt.xlabel( x_l )
        plt.ylabel( y2_l )  
        plt.plot(x, y2, self.pcolor, markersize = 4 )
        plt.tight_layout()
        plt.ticklabel_format( scilimits=(-3, 4) )
        
        self.fig.canvas.draw()
    
#----------------------------------------------------------------------------------------------------------------------------------------   
    def write_time( 
                self,
                file_path: str,
                name: str,
                time
                ):
    
        text = '#\t{}: {}\n'.format( name, time )
        self.InsertText( text, file_path )
          
#----------------------------------------------------------------------------------------------------------------------------------------   
    def start_time( 
                self,
                file_path: str
                ):
    
        t = datetime.datetime.now()
        self.write_time( file_path, 'Start', t )
#----------------------------------------------------------------------------------------------------------------------------------------   
    def sweep_time( 
                self,
                file_path: str
                ):
    
        t = datetime.datetime.now()
        self.write_time( file_path, 'Sweep', t )        
#----------------------------------------------------------------------------------------------------------------------------------------   
    def end_time( 
                self,
                file_path: str
                ):
    
        t = datetime.datetime.now()
        self.write_time( file_path, 'End', t )       

    def get_time_from_line(self, line):
        """
        Extracts the time from a line in the format,"#    #End: YYYY-MM-DD HH:MM:SS.ssssss"
        handling delimiter lines.
        """
        if line.startswith("#\t") and (line.startswith("#\tStart") or line.startswith("#\tEnd")):
            time_str = line.split(": ")[-1].strip()
            return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
        return None

    def process_file(self, filename):
        """
        Processes the file to keep only the earliest start time and latest end time,
        calculates the time difference, and writes it back to the same file.
        """
        # Read the file
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Initialize variables to store the earliest start time and latest end time
        earliest_start_time = None
        latest_end_time = None
        other_lines = []

        # Iterate over the lines to find the earliest start time and latest end time
        for line in lines:
            time = self.get_time_from_line(line)
            if time:
                if line.startswith("#\tStart"):
                    if earliest_start_time is None or time < earliest_start_time:
                        earliest_start_time = time
                elif line.startswith("#\tEnd"):
                    if latest_end_time is None or time > latest_end_time:
                        latest_end_time = time
            else:
                other_lines.append(line)

        # Calculate time difference
        if earliest_start_time and latest_end_time:
            time_diff = latest_end_time - earliest_start_time

            # Format time difference
            time_diff_str = str(time_diff)

            # Prepare the result to be written back to the same file
            result = [
                f"#\tStart: {earliest_start_time}\n",
                f"#\tEnd: {latest_end_time}\n",
                f"#\tTime Difference: {time_diff_str}\n"
            ]

            # Write everything back to the same file
            with open(filename, 'w') as f:
                f.writelines(result + other_lines)


#----------------------------------------------------------------------------------------------------------------------------------------    
    def load_lake( self, lake ):
        
        self.lake = lake
#----------------------------------------------------------------------------------------------------------------------------------------        
    def load_k3( self, k3 ):
        
        self.k3 = k3
#----------------------------------------------------------------------------------------------------------------------------------------        
    def load_mag( self, mag_x = '', mag_y = '', mag_z = '' ):
        
        self.mag_x = mag_x
        self.mag_y = mag_y
        self.mag_z = mag_z
        
#----------------------------------------------------------------------------------------------------------------------------------------    
    def set_keithley(
                self,
                measure: DC_setup.Measure
                ):
        
        if measure.keithley[ -1 ] == '1':
            #print(self.k1)
            return self.k1
        
        if measure.keithley[ -1 ] == '2':
            
            return self.k2
        
        if measure.keithley[ -1 ] == '3':
            
            return self.k3
#----------------------------------------------------------------------------------------------------------------------------------------
    def set_lockin( 
                    self,
                    s: DC_setup.Lock_OUT,
                    m: DC_setup.Lock_IN
                  ):
        
        self.lockin.oscs[ 0 ].freq( s.frequency )
        
        self.lockin.demods[0].timeconstant( m.tc )
        
        self.lockin.sigouts[ 0 ].enables[ 0 ].value( 1 )
        
        self.lockin.sigouts[ 0 ].amplitudes[ 0 ].value( s.signal )
        
        self.lockin.sigouts[ 0 ].on( True )
        
        sleep( 1 )
        
        return self.lockin
    
    def tc_freq_check(self,
                       s: DC_setup.Lock_OUT, 
                       m: DC_setup.Lock_IN
                       ):
        tc_value = self.lockin.demods[0].timeconstant(m.tc)
        freq_value = 1 / (2 * np.pi * self.lockin.oscs[0].freq(s.frequency))
    
        if tc_value <= freq_value:
        # Prompt the user for confirmation to continue or stop
            while True:
                check = input(f"Warning: Time constant {tc_value}s is less than or equal to the lowest value set by {1/(2*np.pi*freq_value)}s, with set frequency {freq_value}Hz.\nDo you want to continue? (yes/no): ").lower()
                
                if check == 'yes':
                    print("Continuing with the process...")
                    break  # Exit the loop and continue
                elif check == 'no':
                    raise ValueError(f"Process stopped. Time constant {tc_value} is less than or equal to the frequency limit {freq_value}.")
                else:
                    print("Please enter 'yes' or 'no'.") 
    
#----------------------------------------------------------------------------------------------------------------------------------------    
    def voltage_change_per_iteration(self, 
                                    ramp_speed_mV_per_s, 
                                    sleep_time=0.5, 
                                    amplifier_factor=1
                                    ):
        """
        Calculate the voltage change per iteration based on ramp speed, sleep time,
        and amplification factor.
        Say you want a final ramp_speed in mV/s of 2 mV/s with amplifier 5:
        voltage_change_per_iteration(5, 2, 5)
        
        # Example usage:
        # voltage_change_per_iteration(5, 0.5, 1) -> 2.5 iteration speed (per sleep time this is howmuch you go up) 
        # Calculates the voltage change per iteration for a ramp speed of 5 mV/s, sleep time of 0.5s, and amplification factor of 1
        # To be used mainly in functions
        """
        return ramp_speed_mV_per_s * sleep_time / amplifier_factor
    
    def ramp_voltage(self, 
                    dac_index, 
                    target_voltage, 
                    ramp_speed_mV_per_s, 
                    max_polarity = 2, 
                    sleep_time=0.5, 
                    amplifier_factor=1, 
                    silent = False, 
                    warning = True):
        """
        Ramps the gate voltage of a DAC to a target voltage at a specified ramp speed.
        For high ramp speeds (> 20) expect significant delays.

        Parameters:.
        - dac_index: The index of the DAC whose voltage needs to be ramped.
        - target_voltage: The target voltage to ramp the DAC to (in mV). 
                          The amplification will happen on this voltage! (e.g. TV 1V, amp 5x -> goes to 5V)
        - ramp_speed_mV_per_s: The desired ramp speed in millivolts per second. (actual ramp speed you want to have)
        - max_polarity : the polarity you set (= abs(polarity), 2 for bipolar, 4 for unipolar)
        - sleep_time: The duration to sleep between each iteration in seconds (default is 0.5s).
        - amplifier_factor: The amplification factor of the system (default is 1 mV/V)

        Example usage:
        ramp_gate( 4, 10, 5, 0.5, 2)
        # Ramps the gate voltage of DAC 4 to 10 V at a speed of 5 mV/s with an amplification factor of 2
        """
        dac = f'dac{dac_index}'
        if warning:
            if ramp_speed_mV_per_s > 250:
                warning_message = f"Warning: Ramp speed is above 250 mV/s ({ramp_speed_mV_per_s} mV/s)."
                warnings.warn(warning_message)
                response = input("To proceed, type 'OK': ")
                if response.lower() != "ok":
                    print('Reply was not ok: aborting')
                    return  # Exit function if response is not 'OK'
        
        current_voltage = self.ivvi.dac_voltages()[dac_index-1]  # Assuming dac_index is 1-based
        #print(current_voltage)
        min_step = max_polarity / (2**16)  # Smallest step in volts due to 16-bit resolution
        min_voltage = min_step * amplifier_factor  # Minimum voltage considering the amplifier
        voltage_change = self.voltage_change_per_iteration(ramp_speed_mV_per_s, sleep_time, amplifier_factor)

        while abs(current_voltage - target_voltage) > min_voltage:
            if current_voltage < target_voltage:
                current_voltage += voltage_change
                if current_voltage > target_voltage:
                    current_voltage = target_voltage
            else:
                current_voltage -= voltage_change
                if current_voltage < target_voltage:
                    current_voltage = target_voltage
            
            # Set DAC voltage using eval
            self.ivvi.set(dac, current_voltage)
            #eval(f'IVVI.dac{dac_index}({current_voltage})')
            sleep(sleep_time)
            #print(f'IVVI.dac{dac_index}({current_voltage})')

        # Ensure final voltage is set to the target voltage using eval
        self.ivvi.set(dac, target_voltage)
        #eval(f'IVVI.dac{dac_index}({target_voltage})')
        if not silent:
            print(f'DAC{dac_index} is {self.ivvi.dac_voltages()[ dac_index - 1 ]} mV')

    def ramp_voltage_to_zero(self, 
                          dac_index,
                          ramp_speed_mV_per_s,
                          max_polarity = 2,
                          sleep_time=0.5,
                          amplifier_factor=1,
                          silent = False,
                          warning = True):
        """
        Ramps the gate voltage of a DAC to zero at a specified ramp speed.
        For high ramp speeds (> 20) expect significant delays.
        
        Parameters:
        - IVVI: The IVVI rack object.
        - dac_index: The index of the DAC whose voltage needs to be ramped.
        - ramp_speed_mV_per_s: The desired ramp speed in millivolts per second. (note: code is of course slightly slower, so this is a max bound)
        - max_polarity : the polarity you set (= abs(polarity), 2 for bipolar, 4 for unipolar)
        - sleep_time: The duration to sleep between each iteration in seconds (default is 0.5s).
        - amplifier_factor: The amplification factor of the system (default is 1).

        Example usage:
        ramp_gate_to_zero(4, 5, 0.5, 1)
        # Ramps the gate voltage of DAC 4 to near 0 at a speed of 5 mV/s with an amplification factor of 1
        """
        dac = f'dac{dac_index}'
        if warning:
            if ramp_speed_mV_per_s * amplifier_factor > 250:
                warning_message = f"Warning: Ramp speed (inc. amplifier) is above 250 mV/s ({ramp_speed_mV_per_s*amplifier_factor} mV/s)."
                warnings.warn(warning_message)
                response = input("To proceed, type 'OK': ")
                if response.lower() != "ok":
                    print('Reply was not ok: aborting')
                    return  # Exit function if response is not 'OK'
                    
        current_voltage = self.ivvi.dac_voltages()[dac_index-1]  # Assuming dac_index is 1-based
        #print(current_voltage)
        min_step = max_polarity / (2**16)  # Smallest step in volts due to 16-bit resolution
        min_voltage = min_step * amplifier_factor  # Minimum voltage step considering the amplifier
        voltage_change = self.voltage_change_per_iteration(ramp_speed_mV_per_s, sleep_time, amplifier_factor)

        while abs(current_voltage) > min_voltage:
            if current_voltage > 0:
                current_voltage -= voltage_change
                if current_voltage < min_voltage:
                    current_voltage = min_voltage
            else:
                current_voltage += voltage_change
                if current_voltage > -min_voltage:
                    current_voltage = -min_voltage
            
            # Set DAC voltage
            self.ivvi.set(dac, current_voltage)
            #eval(f'IVVI.dac{dac_index}({current_voltage})')
            sleep(sleep_time)
            #print(f'IVVI.dac{dac_index}({current_voltage})')
            
        # Ensure final voltage is set to the target voltage using eval
        self.ivvi.set(dac, 0)
        if not silent:
            print(f'DAC{dac_index} is {self.ivvi.dac_voltages()[ dac_index - 1 ]} mV')
# 
#----------------------------------------------------------------------------------------------------------------------------------------        

#---------------------------------------------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------------------------------------------- 
  
#---------------------------------------------------------------------------------------------------------------------------------------- 
    
    def hidden_2p(
                    self,
                    path: str,
                    dac: str,
                    dac_n: int,
                    keithley: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    sweep: list,
                    s_sleep: float,
                ):

        for s in sweep:

            self.ivvi.set( dac, s )

            sleep( s_sleep )

            s_data = self.ivvi.dac_voltages()[ dac_n - 1 ]
            m_data = keithley.amplitude()

            if self.figure:

                self.draw_figure( s_data, m_data, 'source', 'measure' )

            data = np.column_stack( [ s_data, m_data ] )  
            with open( path, 'a+' ) as f:

                np.savetxt( f, data, delimiter = '\t' ) 
                f.flush()              
#----------------------------------------------------------------------------------------------------------------------------------------                 
    def hidden_2p_lock(
                            self,
                            path: str,
                            dac: str,
                            dac_n: int,
                            keithley: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                            lockin: zhinst.qcodes.driver.devices.base.ZIBaseInstrument,
                            sweep: list,
                            s_sleep: float
                            ):

        for s in sweep:

            self.ivvi.set( dac, s )

            sleep( s_sleep )

            s_data = self.ivvi.dac_voltages()[ dac_n - 1 ]
            ds = lockin.sigouts[ 0 ].amplitudes[ 0 ].value()
            m_data = keithley.amplitude()
            dm_re = lockin.demods[ 0 ].sample()[ 'x' ][ 0 ]
            dm_im = lockin.demods[ 0 ].sample()[ 'y' ][ 0 ] 
            
            if self.figure:

                self.draw_figure_y1y2( s_data, m_data, abs( complex( dm_re, dm_im ) )/ds , 'source', 'measure', 'dm/ds ' )
            
            data = np.column_stack( [ s_data, m_data, ds, dm_re, dm_im ] )  
            with open( path, 'a+' ) as f:

                np.savetxt( f, data, delimiter = '\t' ) 
                f.flush()                
#---------------------------------------------------------------------------------------------------------------------------------------- 
    def hidden_bias_2p(
                    self,
                    path: str,
                    b_dac: str,
                    b_dac_n: int,
                    s_dac: str,
                    s_dac_n: int,
                    keithley: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    sweep: list,
                    b_sleep: float,
                    s_sleep: float,
                    ):
        
        self.ivvi.set( b_dac, s )
        
        sleep( b_sleep )
        
        for s in sweep:

            self.ivvi.set( s_dac, s )

            sleep( s_sleep )

            s_data = self.ivvi.dac_voltages()[ s_dac_n - 1 ]
            b_data = self.ivvi.dac_voltages()[ b_dac_n - 1 ]
            m_data = keithley.amplitude()

            if self.figure:

                self.draw_figure( s_data, m_data, 'source', 'measure' )

            data = np.column_stack( [ b_data, s_data, m_data ] )  
            with open( path, 'a+' ) as f:

                np.savetxt( f, data, delimiter = '\t' ) 
                f.flush()
#---------------------------------------------------------------------------------------------------------------------------------------- 
    def hidden_bias_2p_lock(
                    self,
                    path: str,
                    b_dac: str,
                    b_dac_n: int,
                    s_dac: str,
                    s_dac_n: int,
                    keithley: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    lockin: zhinst.qcodes.driver.devices.base.ZIBaseInstrument,            
                    sweep: list,
                    b_sleep: float,
                    s_sleep: float,
                    ):
        
        self.ivvi.set( b_dac, s )
        
        sleep( b_sleep )
        
        for s in sweep:

            self.ivvi.set( s_dac, s )

            sleep( s_sleep )

            s_data = self.ivvi.dac_voltages()[ s_dac_n - 1 ]
            b_data = self.ivvi.dac_voltages()[ b_dac_n - 1 ]
            ds = lockin.sigouts[ 0 ].amplitudes[ 0 ].value()
            
            m_data = keithley.amplitude()
            dm_re = lockin.demods[ 0 ].sample()[ 'x' ][ 0 ]
            dm_im = lockin.demods[ 0 ].sample()[ 'y' ][ 0 ] 
            
            if self.figure:

                self.draw_figure_y1y2( s_data, m_data, abs( complex( dm_re, dm_im ) )/ds , 'source', 'measure', 'dm/ds ' )
            
            data = np.column_stack( [ b_data, s_data, m_data, ds, dm_re, dm_im ] )  
            with open( path, 'a+' ) as f:

                np.savetxt( f, data, delimiter = '\t' ) 
                f.flush()
#----------------------------------------------------------------------------------------------------------------------------------------                 
    def hidden_4p(
                    self,
                    path: str,
                    dac: str,
                    dac_n: int,
                    keithley1: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    keithley2: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    sweep: list,
                    s_sleep: float,
                    silent = True
                    ):
        '''
        path: path
        dac: used dac (source)
        dacn: used dacn (source)
        sweep: rangeo of sleep
        s_sleep: after every point, how long you wait

        KNOWN BUGS:
        ---
        Future possible additions:
        include maximum sweeping speed
        '''
        for s in sweep:
            if not silent and s == int(round(len(sweep))/2):
                print(f'Halfway there: point {int(round(len(sweep))/2)}/{len(sweep)}')
            self.ivvi.set( dac, s )

            sleep( s_sleep )

            s_data = self.ivvi.dac_voltages()[ dac_n - 1 ]
            m1_data = keithley1.amplitude()
            m2_data = keithley2.amplitude()

            if self.figure:

                self.draw_figure( m1_data, m2_data, 'source_m', 'measure' )

            data = np.column_stack( [ s_data, m1_data, m2_data ] )  
            with open( path, 'a+' ) as f:
                np.savetxt( f, data, delimiter = '\t' ) 
                f.flush()
#----------------------------------------------------------------------------------------------------------------------------------------                 
    def hidden_4p_lock(
                    self,
                    path: str,
                    dac: str,
                    dac_n: int,
                    keithley1: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    keithley2: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    lockin: zhinst.qcodes.driver.devices.base.ZIBaseInstrument,
                    sweep: list,
                    s_sleep: float,
                    ):

        for s in sweep:

            self.ivvi.set( dac, s )

            sleep( s_sleep )

            s_data = self.ivvi.dac_voltages()[ dac_n - 1 ]
            ds = lockin.sigouts[ 0 ].amplitudes[ 0 ].value()
            
            m1_data = keithley1.amplitude()
            m2_data = keithley2.amplitude()
            dm_re = lockin.demods[ 0 ].sample()[ 'x' ][ 0 ]
            dm_im = lockin.demods[ 0 ].sample()[ 'y' ][ 0 ] 

            if self.figure:

                self.draw_figure_y1y2( m1_data, m2_data, abs( complex( dm_re, dm_im ) )/ds,'source_m', 'measure', 'dm/ds' )

            data = np.column_stack( [ s_data, m1_data, m2_data, ds, dm_re, dm_im ] )  
            with open( path, 'a+' ) as f:

                np.savetxt( f, data, delimiter = '\t' ) 
                f.flush() 
#----------------------------------------------------------------------------------------------------------------------------------------                 
    def hidden_bias_4p(
                    self,
                    path: str,
                    b_dac: str,
                    b_dac_n: int,
                    s_dac: str,
                    s_dac_n: int,
                    keithley1: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    keithley2: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    sweep: list,
                    b_sleep: float,
                    s_sleep: float,
                    ):
        
        self.ivvi.set( b_dac, s )
        
        sleep( b_sleep )
        
        for s in sweep:

            self.ivvi.set( s_dac, s )

            sleep( s_sleep )

            s_data = self.ivvi.dac_voltages()[ s_dac_n - 1 ]
            b_data = self.ivvi.dac_voltages()[ b_dac_n - 1 ]
            m1_data = keithley1.amplitude()
            m2_data = keithley2.amplitude()

            if self.figure:

                self.draw_figure( m1_data, m2_data, 'source_m', 'measure' )

            data = np.column_stack( [ b_data, s_data, m1_data, m2_data ] )  
            with open( path, 'a+' ) as f:

                np.savetxt( f, data, delimiter = '\t' ) 
                f.flush()
#----------------------------------------------------------------------------------------------------------------------------------------                 
    def hidden_bias_4p_lock(
                    self,
                    path: str,
                    b_dac: str,
                    b_dac_n: int,
                    s_dac: str,
                    s_dac_n: int,
                    keithley1: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    keithley2: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    lockin: zhinst.qcodes.driver.devices.base.ZIBaseInstrument,        
                    sweep: list,
                    b_sleep: float,
                    s_sleep: float,
                    ):
        
        self.ivvi.set( b_dac, s )
        
        sleep( b_sleep )
        
        for s in sweep:

            self.ivvi.set( s_dac, s )

            sleep( s_sleep )

            s_data = self.ivvi.dac_voltages()[ s_dac_n - 1 ]
            b_data = self.ivvi.dac_voltages()[ b_dac_n - 1 ]
            ds = lockin.sigouts[ 0 ].amplitudes[ 0 ].value()
            m1_data = keithley1.amplitude()
            m2_data = keithley2.amplitude()
            dm_re = lockin.demods[ 0 ].sample()[ 'x' ][ 0 ]
            dm_im = lockin.demods[ 0 ].sample()[ 'y' ][ 0 ] 

            if self.figure:

                self.draw_figure_y1y2( m1_data, m2_data, abs( complex( dm_re, dm_im ) )/ds, 'source_m', 'measure', 'dm/ds' )
                
            data = np.column_stack( [ b_data, s_data, m1_data, m2_data, ds, dm_re, dm_im ] )  
            with open( path, 'a+' ) as f:

                np.savetxt( f, data, delimiter = '\t' ) 
                f.flush()
                               
#---------------------------------------------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------------------------------------------- 
#---------------------------------------------------------------------------------------------------------------------------------------- 
    def m2p(
            self,
            file_path: str,
            s: DC_setup.Source,
            m: DC_setup.Measure,
            s_sleep: float,
            repeat = 0
            ):
        ''' 
        This is a 2-points measurement: one source (current or voltage) and one measure (voltage or current). NO LOCKIN
        You can set a sleeping time after the source reach a value.
        You can change the repeat number if you want to repeat the measurement several times
        '''
        s_dac = s.dac
        s_dacn = m.dacn
        k = self.set_keithley( m )
        sweep = s.sweep_or_bias

        self.start_time( file_path )

        self.hidden_2p( file_path, s_dac, s_dacn, k, sweep, s_sleep )

        if s.sweep_back:

            self.hidden_2p( file_path, s_dac, s_dacn, k, sweep[ ::-1 ], s_sleep )

        if repeat != 0:

            self.sweep_time( file_path )

            for i in range( 0, repeat, 1 ):

                self.hidden_2p( file_path, s_dac, s_dacn, k, sweep, s_sleep )

        self.end_time( file_path )
#---------------------------------------------------------------------------------------------------------------------------------------- 
    def m4p(
            self,
            file_path: str,
            s: DC_setup.Source,
            m_s: DC_setup.Measure,
            m_m: DC_setup.Measure,
            s_sleep: float,
            repeat = 0,
            silent = True
            ):
        '''
        file_path = file path
        s = the voltage/ current source you use
        m_s = measure of source voltage/current (same as applied)
        m_m = measure of what is not applied
        '''
        s_dac = s.dac
        s_dacn = s.dacn
        k_s = self.set_keithley( m_s )
        k_m = self.set_keithley( m_m )
        sweep = s.sweep_or_bias # sweep the source or step the source (DAC has a function)

        self.start_time( file_path )

        self.hidden_4p( file_path, s_dac, s_dacn, k_s, k_m, sweep, s_sleep, silent )

        if s.sweep_back:
            
            self.pcolor = 'ro'
            
            self.hidden_4p( file_path, s_dac, s_dacn, k_s, k_m, sweep[ ::-1 ], s_sleep, silent )
        
        self.pcolor = 'bo'
        
        if repeat != 0:

            self.sweep_time( file_path )

            for i in range( 0, repeat, 1 ):

                self.hidden_4p( file_path, s_dac, s_dacn, k_s, k_m, sweep, s_sleep, silent )

        self.end_time( file_path )
#---------------------------------------------------------------------------------------------------------------------------------------- 
    def m4p_lk(
                self,
                file_path: str,
                s: DC_setup.Source,
                m_s: DC_setup.Measure,
                m_m: DC_setup.Measure,
                lk_s: DC_setup.Lock_OUT,
                lk_m: DC_setup.Lock_IN,
                s_sleep: float,
                repeat = 0
            ):

            s_dac = s.dac
            s_dacn = s.dacn
            k_s = self.set_keithley( m_s )
            k_m = self.set_keithley( m_m )
            lk = self.set_lockin( lk_s, lk_m ) 
            sweep = s.sweep_or_bias
            
            self.start_time( file_path )

            self.hidden_4p_lock( file_path, s_dac, s_dacn, k_s, k_m, lk, sweep, s_sleep )

            if s.sweep_back:

                self.pcolor = 'ro'
                
                self.hidden_4p_lock( file_path, s_dac, s_dacn, k_s, k_m, lk, sweep[ ::-1 ], s_sleep )           
                
            self.pcolor = 'bo'
                
            if repeat != 0:

                self.sweep_time( file_path )

                for i in range( 0, repeat, 1 ):

                    self.hidden_4p_lock( file_path, s_dac, s_dacn, k_s, k_m, lk, sweep, s_sleep )
                    
            self.end_time( file_path )
            
            lk.sigouts[ 0 ].on( False )   
#---------------------------------------------------------------------------------------------------------------------------------------- 

    def bias_2p(
                self,
                file_path: str,
                b: DC_setup.Source,
                s: DC_setup.Source,
                m: DC_setup.Measure,
                b_sleep: float,
                s_sleep: float,
                repeat = 0
            ):

            b_dac = b.dac
            b_dacn = b.dacn
            s_dac = s.dac
            s_dacn = s.dacn
            k = self.set_keithley( m )
            sweep = s.sweep_or_bias
            
            self.start_time( file_path )

            self.hidden_bias_2p( file_path, b_dac, b_dacn, s_dac, s_dacn, k, sweep, b_sleep, s_sleep )

            if s.sweep_back:
                
                self.pcolor = 'ro'

                self.hidden_bias_2p( file_path, b_dac, b_dacn, s_dac, s_dacn, k, sweep[ ::-1 ], b_sleep, s_sleep )
            
            self.pcolor = 'bo' 
            
            if repeat != 0:

                self.sweep_time( file_path )

                for i in range( 0, repeat, 1 ):

                    self.hidden_bias_2p( file_path, b_dac, b_dacn, s_dac, s_dacn, k, sweep, b_sleep, s_sleep )
                    
            self.end_time( file_path )
            
#---------------------------------------------------------------------------------------------------------------------------------------- 

    def bias_2p_lk(
                self,
                file_path: str,
                b: DC_setup.Source,
                s: DC_setup.Source,
                m: DC_setup.Measure,
                lk_s: DC_setup.Lock_OUT,
                lk_m: DC_setup.Lock_IN,
                b_sleep: float,
                s_sleep: float,
                repeat = 0
            ):

            b_dac = b.dac
            b_dacn = b.dacn
            s_dac = s.dac
            s_dacn = s.dacn
            m = self.set_keithley( measure )
            lk = self.set_lockin( lk_s, lk_m )
            sweep = s.sweep_or_bias
            
            self.start_time( file_path )

            self.hidden_bias_2p_lock( file_path, b_dac, b_dacn, s_dac, s_dacn, k, lk, sweep, b_sleep, s_sleep )

            if s.sweep_back:

                self.hidden_bias_2p_lock( file_path, b_dac, b_dacn, s_dac, s_dacn, k, lk, sweep[ ::-1 ], b_sleep, s_sleep )
                
            if repeat != 0:

                self.sweep_time( file_path )

                for i in range( 0, repeat, 1 ):

                    self.hidden_bias_2p_lock( file_path, b_dac, b_dacn, s_dac, s_dacn, k, lk, sweep, b_sleep, s_sleep )
                    
            self.end_time( file_path )
            
            lk.sigouts[ 0 ].on( False )   
#---------------------------------------------------------------------------------------------------------------------------------------- 
    @multimethod
    def bias_4p(
                self,
                file_path: str,
                b: DC_setup.Source,
                s: DC_setup.Source,
                m_s: DC_setup.Measure,
                m_m: DC_setup.Measure,
                b_sleep: float,
                s_sleep: float,
                repeat = 0
            ):

            b_dac = b.dac
            b_dacn = b.dacn
            s_dac = s.dac
            s_dacn = s.dacn
            k_s = self.set_keithley( m_s )
            k_m = self.set_keithley( m_m )           
            sweep = source.sweep_or_bias
            
            self.start_time( file_path )

            self.hidden_bias_4p( file_path, b_dac, b_dacn, s_dac, s_dacn, k_s, k_m, sweep, b_sleep, s_sleep )

            if s.sweep_back:

                self.hidden_bias_4p( file_path, b_dac, b_dacn, s_dac, s_dacn, k_s, k_m, sweep[ ::-1 ], b_sleep, s_sleep )
                
            if repeat != 0:

                self.sweep_time( file_path )

                for i in range( 0, repeat, 1 ):

                    self.hidden_bias_4p( file_path, b_dac, b_dacn, s_dac, s_dacn, k_s, k_m, sweep, b_sleep, s_sleep )
                    
            self.end_time( file_path )
            
#---------------------------------------------------------------------------------------------------------------------------------------- 
    @multimethod
    def bias_4p(
                self,
                file_path: str,
                b: DC_setup.Source,
                s: DC_setup.Source,
                m_s: DC_setup.Measure,
                m_m: DC_setup.Measure,
                lk_s: DC_setup.Lock_OUT,
                lk_m: DC_setup.Lock_IN,
                b_sleep: float,
                s_sleep: float,
                repeat = 0
            ):
            '''
            UNCLEAR FUNCTION!!!
            b: bias (either magnetic field or gate)            
            s: source
            m_s: measure source type (voltage/ current)
            m_m: measure what is not source type 
            lk_s: oscillating source you send with the lockin 
            lk_m: signal you demodulate with locking
            b_sleep: sleep time after each bias point
            s_sleep: sleep time after each source point
            '''
            b_dac = b.dac
            b_dacn = b.dacn
            s_dac = s.dac
            s_dacn = s.dacn
            k_s = self.set_keithley( m_s )
            k_m = self.set_keithley( m_m )
            lk = self.set_lockin( lk_s, lk_m ) 
            sweep = s.sweep_or_bias
            
            self.start_time( file_path )

            self.hidden_bias_4p_lock( file_path, b_dac, b_dacn, s_dac, s_dacn, k_s, k_m, lk, sweep, b_sleep, s_sleep )

            if s.sweep_back:

                self.hidden_bias_4p_lock( file_path, b_dac, b_dacn, s_dac, s_dacn, k_s, k_m, lk, sweep[ ::-1 ], b_sleep, s_sleep )
                
            if repeat != 0:

                self.sweep_time( file_path )

                for i in range( 0, repeat, 1 ):

                    self.hidden_bias_4p_lock( file_path, b_dac, b_dacn, s_dac, s_dacn, k_s, k_m, lk, sweep, b_sleep, s_sleep )
                    
            self.end_time( file_path )
            
            lk.sigouts[ 0 ].on( False )   
            
    def hidden_4p_w_PS(
            self,
            path: str,
            dac: str,
            dac_n: int,
            keithley1: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
            keithley2: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
            sweep: list,
            s_sleep: float,
            gate_voltage: float = None,
            magnetic_field: Tuple[float, float, float] = None,
            silent=True
        ):
        '''
        INPUT:
        path: path
        dac: used dac (source)
        dacn: used dacn (source)
        sweep: rangeo of sleep
        s_sleep: after every point, how long you wait
        gate_voltage: Gate voltage to include in the first column of data
        magnetic_field: Magnetic field values to include in the first three columns of data as a tuple of three floats (Bx, By, Bz)
        silent: Flag to suppress printing progress messages
        
        EXAMPLE:
        # Include only gate voltage
        station.hidden_4p_w_PS(path, dac, dac_n, keithley1, keithley2, sweep, s_sleep, gate_voltage=gate_voltage)
        # Include only magnetic field values
        station.hidden_4p_w_PS(path, dac, dac_n, keithley1, keithley2, sweep, s_sleep, magnetic_field=(Bx, By, Bz))
        # Include both gate voltage and magnetic field values
        station.hidden_4p_w_PS(path, dac, dac_n, keithley1, keithley2, sweep, s_sleep, gate_voltage=gate_voltage, magnetic_field=(Bx, By, Bz))
        
        KNOWN BUGS:
        --------
        Future possible additions:
        include maximum sweeping speed
        '''
        for s in sweep:
            if not silent and s == int(round(len(sweep))/2):
                print(f'Halfway there: point {int(round(len(sweep))/2)}/{len(sweep)}')
            self.ivvi.set(dac, s)

            sleep(s_sleep)

            s_data = self.ivvi.dac_voltages()[dac_n - 1]
            m1_data = keithley1.amplitude()
            m2_data = keithley2.amplitude()

            # Include gate voltage and magnetic field values
            #print(gate_voltage, magnetic_field)
            if gate_voltage is not None and magnetic_field is not None:
                data = np.column_stack([gate_voltage, *magnetic_field, s_data, m1_data, m2_data])
            elif gate_voltage is not None:
                data = np.column_stack([gate_voltage, s_data, m1_data, m2_data])
            elif magnetic_field is not None:
                data = np.column_stack([*magnetic_field, s_data, m1_data, m2_data])
            else:
                data = np.column_stack([s_data, m1_data, m2_data])

            with open(path, 'a+') as f:
                np.savetxt(f, data, delimiter='\t')
                f.flush()
                

    def m4p_w_PS(
            self,
            file_path: str,
            s: DC_setup.Source,
            m_s: DC_setup.Measure,
            m_m: DC_setup.Measure,
            s_sleep: float,
            gate_voltage: float = None,
            magnetic_field: Tuple[float, float, float] = None,
            repeat: int = 0,
            silent: bool = True
            ):
        '''
        INPUT
        measure 4 point with parameter sweep
        file_path = file path
        s = the voltage/ current source you use
        m_s = measure of source current or voltage 
              !! Order has to be same as in setup!!
        m_m = measure of what is not applied
              !! Order has to be same as in setup!!
        s_sleep = time to sleep after each measurement
        gate_voltage = gate voltage to include in the data (default is None)
        magnetic_field = magnetic field values to include in the data as a tuple of three floats (Bx, By, Bz) (default is None)
        repeat = number of repetitions (default is 0)
        silent = flag to suppress printing progress messages (default is True)
        
        EXAMPLE:
        #Include only gate voltage
        station.m4p_w_PS(file_path, voltage_source, measure_current, measure_voltage, s_sleep=0.1, gate_voltage=gate_voltage)
        #Include only magnetic field values
        station.m4p_w_PS(file_path, voltage_source, measure_current, measure_voltage, s_sleep=0.1, magnetic_field=(Bx, By, Bz))
        #Include both gate voltage and magnetic field values
        station.m4p_w_PS(file_path, voltage_source, measure_current, measure_voltage, s_sleep=0.1, gate_voltage=gate_voltage, magnetic_field=(Bx, By, Bz))
        '''
        s_dac = s.dac
        s_dacn = s.dacn
        k_s = self.set_keithley( m_s )
        k_m = self.set_keithley( m_m )
        sweep = s.sweep_or_bias # sweep the source or step the source (DAC has a function)

        self.start_time( file_path )

        self.hidden_4p_w_PS(file_path, s_dac, s_dacn, k_s, k_m, sweep, s_sleep, gate_voltage, magnetic_field, silent)

        if s.sweep_back:
            
            self.pcolor = 'ro'
            
            self.hidden_4p_w_PS( file_path, s_dac, s_dacn, k_s, k_m, sweep[ ::-1 ], s_sleep, gate_voltage, magnetic_field, silent )
        
        self.pcolor = 'bo'
        
        if repeat != 0:

            self.sweep_time( file_path )

            for i in range( 0, repeat, 1 ):

                self.hidden_4p_w_PS( file_path, s_dac, s_dacn, k_s, k_m, sweep, s_sleep, gate_voltage, magnetic_field, silent )

        self.end_time( file_path )


    def hidden_4p_lock_w_PS(
                    self,
                    path: str,
                    dac: str,
                    dac_n: int,
                    keithley1: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    keithley2: qcodes.instrument_drivers.tektronix.Keithley_6500.Keithley_6500,
                    lockin: zhinst.qcodes.driver.devices.base.ZIBaseInstrument,
                    sweep: list,
                    s_sleep: float,
                    gate_voltage: float = None,
                    magnetic_field: Tuple[float, float, float] = None,
                    silent=True
                    ):
        '''
        Hidden 4point measurement using lockin and parameter sweep
        ### CLEAN UP THIS COMMENT!!! COPIED
        INPUT: 
        path: path
        dac: used dac (source)
        dacn: used dacn (source)
        sweep: rangeo of sleep
        s_sleep: after every point, how long you wait
        gate_voltage: Gate voltage to include in the first column of data
        magnetic_field: Magnetic field values to include in the first three columns of data as a tuple of three floats (Bx, By, Bz)
        silent: Flag to suppress printing progress messages
        
        EXAMPLE:
        # Include only gate voltage
        station.hidden_4p_w_PS(path, dac, dac_n, keithley1, keithley2, sweep, s_sleep, gate_voltage=gate_voltage)
        # Include only magnetic field values
        station.hidden_4p_w_PS(path, dac, dac_n, keithley1, keithley2, sweep, s_sleep, magnetic_field=(Bx, By, Bz))
        # Include both gate voltage and magnetic field values
        station.hidden_4p_w_PS(path, dac, dac_n, keithley1, keithley2, sweep, s_sleep, gate_voltage=gate_voltage, magnetic_field=(Bx, By, Bz))
        
        KNOWN BUGS:
        --------
        Future possible additions:
        include maximum sweeping speed'''
       
        for s in sweep:
            if not silent and s == int(round(len(sweep))/2):
                    print(f'Halfway there: point {int(round(len(sweep))/2)}/{len(sweep)}')
            self.ivvi.set( dac, s )
            sleep( s_sleep )

            s_data = self.ivvi.dac_voltages()[ dac_n - 1 ]
            ds = lockin.sigouts[ 0 ].amplitudes[ 0 ].value()
            m1_data = keithley1.amplitude()
            m2_data = keithley2.amplitude()
            dm_re = lockin.demods[ 0 ].sample()[ 'x' ][ 0 ]
            dm_im = lockin.demods[ 0 ].sample()[ 'y' ][ 0 ] 

#            if self.figure:
#               self.draw_figure_y1y2( m1_data, m2_data, abs( complex( dm_re, dm_im ) )/ds,'source_m', 'measure', 'dm/ds' )
            if gate_voltage is not None and magnetic_field is not None:
                data = np.column_stack([gate_voltage, *magnetic_field,  s_data, m1_data, m2_data, ds, dm_re, dm_im ])
            elif gate_voltage is not None:
                data = np.column_stack([gate_voltage,  s_data, m1_data, m2_data, ds, dm_re, dm_im ])
            elif magnetic_field is not None:
                data = np.column_stack([*magnetic_field,  s_data, m1_data, m2_data, ds, dm_re, dm_im ])
            else:
                data = np.column_stack([ s_data, m1_data, m2_data, ds, dm_re, dm_im ])

            with open( path, 'a+' ) as f:
                np.savetxt( f, data, delimiter = '\t' ) 
                f.flush() 
                
                
    def m4p_lk_w_PS(
                self,
                file_path: str,
                s: DC_setup.Source,
                m_s: DC_setup.Measure,
                m_m: DC_setup.Measure,
                lk_s: DC_setup.Lock_OUT,
                lk_m: DC_setup.Lock_IN,
                s_sleep: float,
                gate_voltage: float = None,
                magnetic_field: Tuple[float, float, float] = None,
                repeat: int = 0,
                safety_on: bool = True,
                silent: bool = True
            ):
        '''
        INPUT
        measure 4 point with parameter sweep
        file_path = file path
        s = the voltage/ current source you use
        m_s = measure of source current or voltage 
              !! Order has to be same as in setup!!
        m_m = measure of what is not applied
              !! Order has to be same as in setup!!
        t_sleep = time to sleep after each measurement
        gate_voltage = gate voltage to include in the data (default is None)
        magnetic_field = magnetic field values to include in the data as a tuple of three floats (Bx, By, Bz) (default is None)
        repeat = number of repetitions (default is 0)
        safety_on: right now checks for the safety of tc >> 1/f in lockin
        silent = flag to suppress printing progress messages (default is True)
        
        EXAMPLE:
        #Include only gate voltage
        station.m4p_w_PS(file_path, voltage_source, measure_current, measure_voltage, t_sleep=0.1, gate_voltage=gate_voltage)
        #Include only magnetic field values
        station.m4p_w_PS(file_path, voltage_source, measure_current, measure_voltage, t_sleep=0.1, magnetic_field=(Bx, By, Bz))
        #Include both gate voltage and magnetic field values
        station.m4p_w_PS(file_path, voltage_source, measure_current, measure_voltage, t_sleep=0.1, gate_voltage=gate_voltage, magnetic_field=(Bx, By, Bz))
        '''
        s_dac = s.dac
        s_dacn = s.dacn
        k_s = self.set_keithley( m_s )
        k_m = self.set_keithley( m_m )
        lk = self.set_lockin( lk_s, lk_m ) 
        sweep = s.sweep_or_bias
        
        if safety_on:
        # Test whether tc >> 1/ 2*pi*freq
            self.tc_freq_check(lk_s, lk_m)
            
        self.start_time( file_path )
        self.hidden_4p_lock_w_PS( file_path, s_dac, s_dacn, k_s, k_m, lk, sweep, s_sleep, gate_voltage, magnetic_field, silent ) #fill in

        if s.sweep_back:
            # This one wont work - fix it for hidden 4p first!
            self.pcolor = 'ro'
            
            self.hidden_4p_lock_w_PS( file_path, s_dac, s_dacn, k_s, k_m, lk, sweep[ ::-1 ], s_sleep, gate_voltage, magnetic_field, silent )           
            
        self.pcolor = 'bo'
            
        if repeat != 0:

            self.sweep_time( file_path )

            for i in range( 0, repeat, 1 ):

                self.hidden_4p_lock_w_PS( file_path, s_dac, s_dacn, k_s, k_m, lk, sweep, s_sleep, gate_voltage, magnetic_field, silent )
                
        self.end_time( file_path )
        
        lk.sigouts[ 0 ].on( False )   
#----------------------------------------------------------------------------------------------------------------------------------------  
#----------------------------------------------------------------------------------------------------------------------------------------  
#----------------------------------------------------------------------------------------------------------------------------------------  
#----------------------------------------------------------------------------------------------------------------------------------------  
#----------------------------------------------------------------------------------------------------------------------------------------  
