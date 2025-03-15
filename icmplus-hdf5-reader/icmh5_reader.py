import numpy as np
import h5py
from dataclasses import dataclass
import datetime
from typing import Generator, List, Tuple, Optional

MICROSEC_IN_SEC = 1_000_000
INVALID_VALUE = -99999


@dataclass
class Interval:
    """Base class for all interval types"""
    start_time: int
    end_time: int
    
    def overlaps(self, other: 'Interval') -> bool:
        return self.start_time < other.end_time and other.start_time < self.end_time

    def intersection(self, other: 'Interval') -> Optional['Interval']:
        if not self.overlaps(other):
            return None
        return Interval(
            start_time=max(self.start_time, other.start_time),
            end_time=min(self.end_time, other.end_time)
        )

@dataclass
class AdvancedInterval(Interval):
    """Interval containing actual data information"""
    start_sample: int
    sample_count: int
    is_good: bool = True
    
    @property
    def end_sample(self) -> int:
        return self.start_sample + self.sample_count
    
@dataclass
class DataSegment:
    values: np.ndarray
    start_time: int  # microseconds
    sampling_freq: float
    
    @property
    def start_datetime(self) -> datetime.datetime:
        """Human-readable start time in UTC"""
        return microseconds_to_datetime(self.start_time)

class IntervalHandler:
    """Base class for handling intervals"""
    def get_intervals(self, start_time: int, end_time: int) -> List[Interval]:
        raise NotImplementedError
    
    @staticmethod
    def intersect_intervals(intervals_a: List[Interval], intervals_b: List[Interval]) -> List[Interval]:
        """Compute intersections between two lists of intervals"""
        result = []
        i = j = 0
        
        while i < len(intervals_a) and j < len(intervals_b):
            intersection = intervals_a[i].intersection(intervals_b[j])
            if intersection:
                result.append(intersection)
            
            if intervals_a[i].end_time < intervals_b[j].end_time:
                i += 1
            else:
                j += 1
                
        return result

class IndexIntervalHandler(IntervalHandler):
    def __init__(self, index_table: np.ndarray):
        self.table = index_table
        if len(self.table) == 0:
            raise ValueError("Index table is empty")
        if not all(self.table['frequency'] == self.table[0]['frequency']):
            raise ValueError("Varying frequencies in index table are not supported")
        self.sampling_freq = self.table[0]['frequency']
        
    def get_intervals(self, start_time: int, end_time: int) -> List[AdvancedInterval]:
        intervals = []
        
        for entry in self.table:
            interval_end = entry['starttime'] + (entry['length'])/entry['frequency']*MICROSEC_IN_SEC
            
            # >= -> in case `start_time` == `intervel_end` 
            if interval_end >= start_time and entry['starttime'] < end_time: 
                intervals.append(AdvancedInterval(
                    start_time=entry['starttime'],
                    end_time=interval_end,
                    start_sample=entry['startidx'],
                    sample_count=entry['length'],
                ))
                
        return intervals
    
    def time_to_sample(self, timestamp: int) -> Tuple[int, int]:
        """Convert timestamp to (sample_index, time_delta)"""
        for interval in self.get_intervals(timestamp, timestamp + 1):
            if interval.start_time <= timestamp < interval.end_time+1:
                offset = (timestamp - interval.start_time) / MICROSEC_IN_SEC
                sample = interval.start_sample + int(offset * self.sampling_freq)
                residual_s = (timestamp - self.sample_to_time(sample))/MICROSEC_IN_SEC
                return sample, residual_s
        return -1, 0
    
    def sample_to_time(self, sample: int) -> int:
        """Convert sample index to microseconds timestamp"""
        for entry in self.table:
            if entry['startidx'] <= sample < entry['startidx'] + entry['length']:
                offset = (sample - entry['startidx']) / entry['frequency']
                return entry['starttime'] + int(offset * MICROSEC_IN_SEC)
        return self.table[-1]['starttime'] + int(self.table[-1]['length']/self.table[-1]['frequency']*MICROSEC_IN_SEC)

class QualityIntervalHandler(IntervalHandler):
    def __init__(self, quality_table: np.ndarray, end_time: int):
        self.quality_table = quality_table
        self.end_time = end_time
        
    def get_intervals(self, start_time: int, end_time: int) -> List[AdvancedInterval]:
        if len(self.quality_table) == 0:
            return [AdvancedInterval(
                start_time=start_time,
                end_time=end_time,
                start_sample=0,
                sample_count=0,
                is_good=True
            )]
            
        intervals = []
        current_time = start_time
        current_is_good = True
        
        for entry in self.quality_table:
            if entry['time'] > current_time:
                intervals.append(AdvancedInterval(
                    start_time=current_time,
                    end_time=min(entry['time'], end_time),
                    start_sample=0,  
                    sample_count=0,
                    is_good=current_is_good
                ))
            current_time = entry['time']
            current_is_good = entry['value'] == 0
            
            if current_time >= end_time:
                break
                
        if current_time < end_time:
            intervals.append(AdvancedInterval(
                start_time=current_time,
                end_time=end_time,
                start_sample=0,
                sample_count=0,
                is_good=current_is_good
            ))
            
        return intervals

def microseconds_to_datetime(microseconds: int) -> datetime.datetime:
    """Convert microseconds since epoch to datetime object"""
    return datetime.datetime.fromtimestamp(microseconds / MICROSEC_IN_SEC, tz=datetime.timezone.utc)


class HDF5SignalReader:
    def __init__(self, hdf5_file: h5py.File, signal_name: str, icmh5_mode='no_sep'):
        if signal_name not in hdf5_file['waves']:
            raise ValueError(f"Signal '{signal_name}' not found in 'waves' group")
        self.signal = hdf5_file['waves'][signal_name]
        self._init_icmh5_mode(icmh5_mode, signal_name, hdf5_file)
        self.index_handler = IndexIntervalHandler(self.index_origin)
        total_duration = self._calculate_total_duration()
        self.quality_handler = QualityIntervalHandler(
            self.quality_origin,
            total_duration
        )
        self.sampling_freq = self.index_handler.sampling_freq
        
    def _init_icmh5_mode(self, icmh5_mode: str, signal_name: str, hdf5_file: h5py.File):
        if icmh5_mode == 'no_sep':
            self.index_origin = self.signal.attrs['index']
            self.quality_origin = self.signal.attrs['quality']
        elif icmh5_mode == 'sep':
            self.index_origin = hdf5_file['waves'][f'{signal_name}.index']
            self.quality_origin = hdf5_file['waves'][f'{signal_name}.quality']

    def _calculate_total_duration(self) -> int:
        last_entry = self.index_handler.table[-1]
        return last_entry['starttime'] + int(last_entry['length']/last_entry['frequency']*MICROSEC_IN_SEC)
    
    def _get_intervals(self, start_time: int, end_time: int, quality_mode: str='good') -> List[Interval]:
        """
           #TODO: Another Simple method
           - no need to write index_handler.get_intervals and quality_handler.get_intervals
           - get quality_handler, bad sections
           - write an exclude method in IntervalHandler
           - set input_intervals = [Interval(start_time, end_time)]
           - final_intervals = Intersect(Exclude(index_intervals, bad_intervals), input_intervals)
        """
        
        if quality_mode not in ['good', 'bad', 'ignore']:
            raise ValueError("quality_mode must be 'good', 'bad', or 'ignore'")
        
        index_intervals = self.index_handler.get_intervals(start_time, end_time)
        
        if quality_mode == 'ignore':
            final_intervals = index_intervals
        else:
            quality_intervals = self.quality_handler.get_intervals(start_time, end_time)
            quality_intervals = [
                interval for interval in quality_intervals
                if interval.is_good == (quality_mode == 'good')
            ]
            
            final_intervals = self.index_handler.intersect_intervals(index_intervals, quality_intervals)
        
        input_interval = [Interval(start_time, end_time)]
        final_intervals = self.index_handler.intersect_intervals(input_interval, final_intervals)
        
        return final_intervals
    
    def _iterate_intervals_to_segments(self, intervals: List[AdvancedInterval]) -> Generator[DataSegment, None, None]:
        for interval in intervals:
            # Clip to requested time range
            seg_start, seg_end = interval.start_time, interval.end_time
            
            # Convert to samples
            start_sample, _ = self.index_handler.time_to_sample(seg_start)
            end_sample, _ = self.index_handler.time_to_sample(seg_end)
            
            # Load data from HDF5
            data = self.signal[start_sample:end_sample+1].astype(float)
            data[data == INVALID_VALUE] = np.nan
            
            yield DataSegment(
                values=data,
                start_time=self.index_handler.sample_to_time(start_sample),
                sampling_freq=self.sampling_freq
            )

    def iterate_segments(self, start_time: int, end_time: int, quality_mode: str='good') -> Generator[DataSegment, None, None]:
        """Generator yielding valid data segments in time range"""
        intervals = self._get_intervals(start_time, end_time, quality_mode)
        yield from self._iterate_intervals_to_segments(intervals)
        
class SignalProcessor:
    def __init__(self, hdf5_path: str, icmh5_mode='no_sep'):
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.reader = None
        self.current_signal = None
        self.current_info = None
        self.icmh5_mode = icmh5_mode

    def get_available_signals(self) -> List[str]:
        sig_list = list(self.hdf5_file['waves'].keys())
        valid_list = []
        for sig in sig_list:
            if 'index' not in sig and 'quality' not in sig:
                valid_list.append(sig)
            else:
                if self.icmh5_mode == 'no_sep':
                    raise ValueError("Signal names should not contain 'index' or 'quality' in 'no_sep' mode")
        return valid_list
    
    def set_signal(self, signal_name: str):
        if signal_name not in self.get_available_signals():
            raise ValueError(f"Signal '{signal_name}' not available")
        self.current_signal = signal_name
        self.reader = HDF5SignalReader(self.hdf5_file, signal_name, self.icmh5_mode)
        
        sampling_freq = self.reader.sampling_freq
        duration = (self.reader._calculate_total_duration() - self.reader.index_handler.table[0]['starttime'])/MICROSEC_IN_SEC
        
        info_dict = {
            'signal_name': signal_name,
            'start_time': self.reader.index_handler.table[0]['starttime'],
            'end_time': self.reader._calculate_total_duration(),
            'sampling_freq': sampling_freq,
            'start_datetime': microseconds_to_datetime(self.reader.index_handler.table[0]['starttime']),
            'end_datetime': microseconds_to_datetime(self.reader._calculate_total_duration()),
            'duration [s]': duration,
            '# of segments': len(self.reader.index_handler.table),
            '# of points': int(duration*sampling_freq),
            
        }
        self.current_info = info_dict

    def load_segments(self, start_time: int, duration: int, quality_mode: str = "good") -> List[DataSegment]:
        """
            Load all continuous data segments (but does not mean no NaN) in the specified time range.
                start_time: start time in microseconds
                duration: duration in microseconds
                quality_mode: 'good', 'bad', or 'ignore'
        """
        if not self.reader:
            raise ValueError("No signal selected. Call set_signal() first.")
        end_time = start_time + duration
        segments = list(self.reader.iterate_segments(start_time, end_time, quality_mode))
        return segments
        
    def load_data(self, start_time: int, duration: int, quality_mode: str ='good') -> DataSegment:
        """
            Get complete data stream with NaN-filled gaps
                start_time: start time in microseconds
                duration: duration in microseconds
                quality_mode: 'good', 'bad', or 'ignore'
            
        """
        if not self.reader:
            raise ValueError("No signal selected. Call set_signal() first.")
        
        end_time = start_time + duration
        num_of_points = int(duration/MICROSEC_IN_SEC * self.reader.sampling_freq)
        result = np.empty(num_of_points)
        result.fill(np.nan)
        
        current_pos = 0
        for segment in self.reader.iterate_segments(start_time, end_time, quality_mode):
            # Calculate position in result array
            start_offset = int((segment.start_time - start_time)/MICROSEC_IN_SEC * self.reader.sampling_freq)
            end_offset = start_offset + len(segment.values)
            
            # Handle boundary conditions
            if end_offset > len(result):
                end_offset = len(result)
                segment.values = segment.values[:end_offset-start_offset]
            
            result[start_offset:end_offset] = segment.values
            current_pos = end_offset
        
        return DataSegment(
            values=result,
            start_time=start_time,
            sampling_freq=self.reader.sampling_freq
        )
    
    def close(self):
        self.hdf5_file.close()

# Example usage
if __name__ == "__main__":
    processor = SignalProcessor("data.hdf5", "icp")
    try:
        data = processor.load_data(1637232715964166, 3600*MICROSEC_IN_SEC)  # 1 hour
        print(f"Retrieved {len(data.values)} samples")
        print(f"First value: {data.values[0]}, Last value: {data.values[-1]}")
    finally:
        processor.close()