import numpy as np
from osgeo import gdal
import time

gdal.DontUseExceptions()

EARTH_RADIUS = 6371000 * 4 / 3
CURVE_CONSTANT = 1 / (2 * EARTH_RADIUS)

class Terrain:
    def __init__(self, file_path: str, scale: float, up = 0):
        """Loads and stores terrain data from a geotiff file
        
        Parameters
        ----------
        file_path: str
            Absolute file path to terrain geotiff
        scale: float
            Sample intervals in meters
            
        Returns
        -------
        Terrain:
            Terrain object"""
        
        self.scale = scale

        # Load terrain data
        ds = gdal.Open(file_path)
        self.terrain = ds.GetRasterBand(1).ReadAsArray().transpose()
        self.terrain += up

    def get_terrain(self) -> np.ndarray:
        """Get terrain data"""
        return self.terrain.transpose()

    def interp_terrain(self, pos: np.ndarray) -> float:
        """Bilinear interpolation between known terrain elevations"""

        pos = pos / self.scale

        p1 = self.terrain[int(np.floor(pos[0])), int(np.floor(pos[1]))]
        p2 = self.terrain[int(np.ceil(pos[0])), int(np.floor(pos[1]))]
        p3 = self.terrain[int(np.floor(pos[0])), int(np.ceil(pos[1]))]
        p4 = self.terrain[int(np.ceil(pos[0])), int(np.ceil(pos[1]))]

        interp1 = (pos[0] - np.floor(pos[0])) * (p2 - p1) + p1
        interp2 = (pos[0] - np.floor(pos[0])) * (p4 - p3) + p3

        return (pos[1] - np.floor(pos[1])) * (interp2 - interp1) + interp1
    
    def earth_drop(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate the drop due to earth curvature"""

        dis_squared = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
        earth_curve = CURVE_CONSTANT * dis_squared

        return earth_curve

class Point:
    def __init__(self, position: list[float], height: float, relative_terrain=True):
        """Simple class storing position in 3D space
        
        Parameters
        ----------
        position: list[float]
            Position on ground
        height: float
            Height above terrain (default) or absolute elevation
        relative_terrain: bool
            Determines whether height is relative to the terrain or absolute
            
        Returns
        -------
        Point:
            Point object"""
    
        if len(position) != 2:
            raise ValueError ("Position must be of length 2")

        self.pos = np.array(position)
        self.h = height

        self.relative_terrain = relative_terrain

class Radar(Point):
    def __init__(self, terrain: Terrain, surface: Terrain, position: list[float], height: float, frequency: float, PRF: float, pulse_length: float, transmitting_power: float, gain: float, water_density: float, system_temperature: float, relative_terrain=True):
        """Class representing a radar transmiter / receiver
        
        Parameters
        ----------
        terrain: Terrain
            Terrain object
        position: list[float]
            Position of the radar on the ground
        height: float
            Height above terrain (default) or absolute elevation in meters
        frequency: float
            Radio frequency used by the radar in Hz
        PRF: float
            Pulse repetition frequency of the radar system in Hz
        pulse_length: float
            Time of each radar pulse in seconds
        transmitting_power: float
            Transmission power in watts
        gain: float
            Gain of the radar transmitter / receiver
        losses: float
            System losses (1 for no losses)
        system_temperature: float
            System temperature to model thermal losses in K
        relative_terrain: bool
            Determines whether height is relative to the terrain or absolute
            
        Returns
        -------
        Radar:
            Radar object"""
        
        super().__init__(position, height, relative_terrain=relative_terrain)
        
        self.terrain = terrain
        self.surface = surface
        self.frequency = frequency
        self.PRF = PRF
        self.pulse_length = pulse_length
        self.power = transmitting_power
        self.gain = gain
        self.system_temperature = system_temperature
        self.minimum_snr_db = -1

        f_ghz = self.frequency / 1e9

        oxygen_factor = 0.001 * f_ghz ** 2 * (0.00719 + 6.09 / (f_ghz ** 2 + 0.227) + 4.81 / ((f_ghz - 57) ** 2 + 1.5))
        water_factor = 0.0001 * f_ghz ** 2 * water_density * (0.05 + 0.0021 * water_density * 3.6 / ((f_ghz - 22.2) ** 2 + 8.5) + 10.6 / ((f_ghz - 183.3) ** 2 + 9) + 8.9 / ((f_ghz - 325.4) ** 2 + 26.3))
        self.atmos_attenuation_coef = oxygen_factor + water_factor

        self.veg_attenuation_coef = 2 * (0.135 * f_ghz + 0.0506)

        self.snr_multiplier = (transmitting_power * gain ** 2 * self.get_wavelength() ** 2 * pulse_length) / ((4 * np.pi) ** 3 * (1.38e-23) * system_temperature)
        
    def get_wavelength(self) -> float:
        return 3e8 / self.frequency
    
    def sample_terrain(self, terrain: Terrain, pos: np.ndarray) -> float:
        """Sample the terrain elevation at a given point"""

        return terrain.interp_terrain(pos) - terrain.earth_drop(self.pos, pos)
    
    def calculate_losses(self, vegetation_distance: float, distance: float) -> float:
        atmos_attenuation = self.atmos_attenuation_coef * distance / 100
        veg_attenuation = self.veg_attenuation_coef * vegetation_distance

        return 10 ** ((atmos_attenuation + veg_attenuation) / 10)
    
    def calculate_snr(self, target: Point, vegetation_distance: float, rcs: float) -> float:
        """Calculate the theoretical signal for the returns from a target
        
        Parameters
        ----------
        target: Point
            Target to calculate the SNR to
        rcs: float
            The radar cross section of the target
            
        Returns
        -------
        float:
            Theoretical SNR of the target (dB)"""
        
        # Calculate the absolute target and radar elevation
        target_elevation = target.h - self.terrain.earth_drop(self.pos, target.pos)
        if target.relative_terrain:
            target_elevation += self.terrain.interp_terrain(target.pos)

        radar_elevation = self.h
        if self.relative_terrain:
            radar_elevation += self.terrain.interp_terrain(self.pos)

        # Calculate the distance between the radar and the target
        distance = np.sqrt((self.pos[0] - target.pos[0]) ** 2 + (self.pos[1] - target.pos[1]) ** 2 + (self.h - target_elevation) ** 2)

        # Calculate the signal to noise ratio
        snr = self.snr_multiplier * rcs / (distance ** 4 * self.calculate_losses(vegetation_distance, distance))

        return 10 * np.log10(snr)
    
    def update_minimum_snr(self, detection_probability: float, false_alarm_probability: float, number_of_samples: int) -> None:
        """Update the minimum snr required for the radar to detect a target
        
        Parameters
        ----------
        detection_probability: float
            The probability of detection when receiving a radar return from an object
        false_alarm_probability: float
            The probability of detection when no radar return is received
        number_of_samples: int
            The number of samples used to determing whether received signals are from a target
            
        Returns
        -------
        None:
            None"""

        if not (0 <= detection_probability <= 1) or not(0 <= false_alarm_probability <= 1):
            raise ValueError("Probabilities must be between 0 and 1 (inclusive)")
        
        if not (0.1 <= detection_probability <= 0.9):
            print("WARNING::For accurate results, detection probability should be between 0.1 and 0.9 (inclusive)")

        if not (1e-7 <= false_alarm_probability <= 1e-3):
            print("WARNING::For accurate results, false_alarm_probability should be between 1e-7 and 1e-3 (inclusive)")
        
        if not (1 <= number_of_samples <= 8096):
            print("WARNING::For accurate results, number of samples should be between 1 and 8096 (inclusive)")

        # Calculate the minimum snr in decibels
        a = np.log(0.62 / false_alarm_probability)
        b = np.log(detection_probability / (1 - detection_probability))
        
        self.minimum_snr_db = -5 * np.log10(number_of_samples) + (6.2 + 4.54 / np.sqrt(number_of_samples + 0.44)) * np.log10(a + 0.12 * a * b + 1.7 * b)
    
    def target_snr(self, target: Point, rcs: float, step_distance=1) -> float:
        """Return boolean representing whether a point is visible to the radar"""

        # Exit if target is radar
        if np.array_equal(self.pos, target.pos) and self.h == target.h:
            return True

        # Set up initial position
        pos = np.zeros(3)
        pos[0:2] = self.pos
        if self.relative_terrain:
            pos[2] = self.terrain.interp_terrain(self.pos) + self.h
        else:
            pos[2] = self.h

        # Set up end position
        end_pos = np.zeros(3)
        end_pos[0:2] = target.pos
        if target.relative_terrain:
            end_pos[2] = self.sample_terrain(self.terrain, target.pos) + target.h
        else:
            end_pos[2] = target.h - self.terrain.earth_drop(self.pos, target.pos)

        # Calculate the step vector
        step = end_pos - pos
        step *= step_distance / np.linalg.norm(step)

        # Calculate the total number of steps required to reach the target
        try:
            total_steps = int(np.floor((end_pos[0] - pos[0]) / step[0])) - 1
        except ValueError:
            try:
                total_steps = int(np.floor((end_pos[1] - pos[1]) / step[1])) - 1
            except ValueError:
                total_steps = int(np.floor((end_pos[2] - pos[2]) / step[2])) - 1

        vegetation_distance = 0

        for _ in range(total_steps):
            # Step along path
            pos += step

            # Check if path is below terrain, NaN if so
            if self.sample_terrain(self.terrain, pos) > pos[2]:
                return np.nan
            
            elif self.sample_terrain(self.surface, pos) > pos[2]:
                vegetation_distance += step_distance
            
        return self.calculate_snr(target, vegetation_distance, rcs)