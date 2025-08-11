'''
Copyright (c) 2019 NeuroNexus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software file and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# NOTE: This code has been modified to allow reading longer xdat files

import os
import json
import numpy as np
from pathlib import Path

MAX_FILE_LOAD_SIZE_BYTES = 1e12   # arbitrary data load limit

def read_allego_xdat_metadata(datasource_name):
    ''' returns data source meta-data as a dictionary

        Parameters
        ----------
        datasource_name : str
            full data source name, including the path & excluding file extensions

        Returns
        -------
        metadata : dict
            all meta-data for the xdat data source

        See also
        --------
        get_allego_xdat_time_range
        read_allego_xdat_all_signals
        read_allego_xdat_pri_signals
        read_allego_xdat_aux_signals
        read_allego_xdat_din_signals
        read_allego_xdat_dout_signals
    '''
    dsource_name = str(Path(datasource_name).expanduser().resolve())
    fname_metadata = '{}.xdat.json'.format(dsource_name)
    try:
        with open(fname_metadata, 'rb') as fid:
            metadata = json.load(fid)
    except Exception as ex:
        raise ValueError('could not load metadata file {}'.format(fname_metadata))
    return metadata

def get_allego_xdat_time_range(datasource_name):
    ''' returns data source time range

        Parameters
        ----------
        datasource_name : str
            full data source name, including the path & excluding file extensions

        Returns
        -------
        metadata : dict
            all meta-data for the xdat data source

        See also
        --------
        read_allego_xdat_metadata
        read_allego_xdat_all_signals
        read_allego_xdat_pri_signals
        read_allego_xdat_aux_signals
        read_allego_xdat_din_signals
        read_allego_xdat_dout_signals
    '''
    return read_allego_xdat_metadata(datasource_name)['status']['t_range']

def read_allego_xdat_all_signals(datasource_name, time_start=None, time_end=None):
    ''' returns data source signal data of all signal types over the requested time range

        Parameters
        ----------
        datasource_name : str
            full data source name, including the path & excluding file extensions
        time_start : float, optional
            requested starting time in seconds
        time_end : float, optional
            requested ending time in seconds

        Returns
        -------
        signal_matrix, timestamps, time_samples : tuple
            signal_matrix : numpy array
                signal data with shape MxN, where M is number of signals and N is number of samples
            timestamps : numpy array
                timestamp data with shape Nx1
            time_samples : numpy array
                time samples (sec) with shape Nx1

        Notes
        -----
        An Allego xdat file contains four types of signals referred to as 'pri', 'aux', 'din', and 'dout'
        This function returns the signal data from all four of these signal types in 'signal_matrix'.

        See also
        --------
        read_allego_xdat_metadata
        get_allego_xdat_time_range
        read_allego_xdat_pri_signals
        read_allego_xdat_aux_signals
        read_allego_xdat_din_signals
        read_allego_xdat_dout_signals
    '''
    metadata = read_allego_xdat_metadata(datasource_name)
    dsource_name = str(Path(datasource_name).expanduser().resolve())
    fname_signal_array = '{}_data.xdat'.format(dsource_name)
    fname_timestamps = '{}_timestamp.xdat'.format(dsource_name)

    fs = metadata['status']['samp_freq']
    time_start = metadata['status']['t_range'][0] if time_start is None else time_start
    time_end = metadata['status']['t_range'][1] if time_end is None else time_end
    num_samples = int(round(time_end * fs)) - int(round(time_start * fs))

    if (num_samples * metadata['status']['signals']['total'] * 4) + (num_samples * 8) > MAX_FILE_LOAD_SIZE_BYTES:
        raise ValueError('requested too large of a dataset to load. Request a smaller time range')

    tstamp_offset = int(round(time_start * fs)) - metadata['status']['timestamp_range'][0]
    if tstamp_offset < 0:
        raise ValueError('requested time start must be >= starting time of file ({})'.format(metadata['status']['t_range'][0]))
    if tstamp_offset + num_samples > Path(fname_timestamps).stat().st_size / 8:
        raise ValueError('requested time end is past the ending time of file ({})'.format(metadata['status']['t_range'][1]))

    try:
        with open(Path(fname_timestamps), 'rb') as fid:
            fid.seek(tstamp_offset * 8, os.SEEK_SET)
            timestamps = np.fromfile(fid, dtype=np.int64, count=num_samples)
    except Exception as ex:
        raise ValueError('could not load timestamp data from file {} : {}'.format(fname_timestamps, ex))

    try:
        with open(Path(fname_signal_array), 'rb') as fid:
            fid.seek(tstamp_offset * metadata['status']['signals']['total'] * 4, os.SEEK_SET)
            raw_sig_array = np.fromfile(fid, dtype=np.float32, count=num_samples * metadata['status']['signals']['total'])
    except Exception as ex:
        raise ValueError('could not load signal array data from file {} : {}'.format(fname_timestamps, ex))

    # basic check of data integrity
    num_samples_sig_array = int(raw_sig_array.size/metadata['status']['signals']['total'])
    if timestamps.shape[0] != num_samples_sig_array:
        raise RuntimeError('inconsistent number of samples between timestamps and signal array')

    return np.reshape(raw_sig_array, (num_samples_sig_array,metadata['status']['signals']['total'])).T, timestamps,  timestamps/fs


def read_allego_xdat_pri_signals(datasource_name, time_start=None, time_end=None):
    ''' returns data source signal data of 'pri' (amplifier) signals over the requested time range

        Parameters
        ----------
        datasource_name : str
            full data source name, including the path & excluding file extensions
        time_start : float, optional
            requested starting time in seconds
        time_end : float, optional
            requested ending time in seconds

        Returns
        -------
        signal_matrix, timestamps, time_samples : tuple
            signal_matrix : numpy array
                signal data with shape MxN, where M is number of primary signals and N is number of samples
            timestamps : numpy array
                timestamp data with shape Nx1
            time_samples : numpy array
                time samples (sec) with shape Nx1

        Notes
        -----
        An Allego xdat file contains four types of signals referred to as 'pri', 'aux', 'din', and 'dout'
        This function returns the 'pri' (amplifier) signal data in 'signal_matrix'.

        See also
        --------
        read_allego_xdat_metadata
        get_allego_xdat_time_range
        read_allego_xdat_all_signals
        read_allego_xdat_aux_signals
        read_allego_xdat_din_signals
        read_allego_xdat_dout_signals
    '''
    sig_array, timestamps, time_samples = read_allego_xdat_all_signals(datasource_name, time_start=time_start, time_end=time_end)
    return sig_array[0:sig_array.shape[0]-6,:], timestamps, time_samples


def read_allego_xdat_aux_signals(datasource_name, time_start=None, time_end=None):
    ''' returns data source signal data of 'aux' (auxillary) signals over the requested time range

        Parameters
        ----------
        datasource_name : str
            full data source name, including the path & excluding file extensions
        time_start : float, optional
            requested starting time in seconds
        time_end : float, optional
            requested ending time in seconds

        Returns
        -------
        signal_matrix, timestamps, time_samples : tuple
            signal_matrix : numpy array
                signal data with shape MxN, where M is number of auxillary signals and N is number of samples
            timestamps : numpy array
                timestamp data with shape Nx1
            time_samples : numpy array
                time samples (sec) with shape Nx1

        Notes
        -----
        An Allego xdat file contains four types of signals referred to as 'pri', 'aux', 'din', and 'dout'
        This function returns the 'aux' signal data in 'signal_matrix'.

        See also
        --------
        read_allego_xdat_metadata
        get_allego_xdat_time_range
        read_allego_xdat_all_signals
        read_allego_xdat_pri_signals
        read_allego_xdat_din_signals
        read_allego_xdat_dout_signals
    '''
    sig_array, timestamps, time_samples = read_allego_xdat_all_signals(datasource_name, time_start=time_start, time_end=time_end)
    aux_start_idx = sig_array.shape[0]-6
    return sig_array[aux_start_idx:aux_start_idx+2, :], timestamps, time_samples

def read_allego_xdat_din_signals(datasource_name, time_start=None, time_end=None):
    ''' returns data source signal data of 'din' (digital input) signals over the requested time range

        Parameters
        ----------
        datasource_name : str
            full data source name, including the path & excluding file extensions
        time_start : float, optional
            requested starting time in seconds
        time_end : float, optional
            requested ending time in seconds

        Returns
        -------
        signal_matrix, timestamps, time_samples : tuple
            signal_matrix : numpy array
                signal data with shape MxN, where M is number of digital in signals and N is number of samples
            timestamps : numpy array
                timestamp data with shape Nx1
            time_samples : numpy array
                time samples (sec) with shape Nx1

        Notes
        -----
        An Allego xdat file contains four types of signals referred to as 'pri', 'aux', 'din', and 'dout'
        This function returns the 'din' signal data in 'signal_matrix'.

        See also
        --------
        read_allego_xdat_metadata
        get_allego_xdat_time_range
        read_allego_xdat_all_signals
        read_allego_xdat_pri_signals
        read_allego_xdat_aux_signals
        read_allego_xdat_dout_signals
    '''
    sig_array, timestamps, time_samples = read_allego_xdat_all_signals(datasource_name, time_start=time_start, time_end=time_end)
    din_start_idx = sig_array.shape[0]-4
    return sig_array[din_start_idx:din_start_idx+2, :], timestamps, time_samples

def read_allego_xdat_dout_signals(datasource_name, time_start=None, time_end=None):
    ''' returns data source signal data of 'dout' (digital out) signals over the requested time range

        Parameters
        ----------
        datasource_name : str
            full data source name, including the path & excluding file extensions
        time_start : float, optional
            requested starting time in seconds
        time_end : float, optional
            requested ending time in seconds

        Returns
        -------
        signal_matrix, timestamps, time_samples : tuple
            signal_matrix : numpy array
                signal data with shape MxN, where M is number of digital out signals and N is number of samples
            timestamps : numpy array
                timestamp data with shape Nx1
            time_samples : numpy array
                time samples (sec) with shape Nx1

        Notes
        -----
        An Allego xdat file contains four types of signals referred to as 'pri', 'aux', 'din', and 'dout'
        This function returns the 'dout' signal data in 'signal_matrix'.

        See also
        --------
        read_allego_xdat_metadata
        get_allego_xdat_time_range
        read_allego_xdat_all_signals
        read_allego_xdat_pri_signals
        read_allego_xdat_aux_signals
        read_allego_xdat_din_signals
    '''
    sig_array, timestamps, time_samples = read_allego_xdat_all_signals(datasource_name, time_start=time_start, time_end=time_end)
    dout_start_idx = sig_array.shape[0]-2
    return sig_array[dout_start_idx:dout_start_idx+2, :], timestamps, time_samples
