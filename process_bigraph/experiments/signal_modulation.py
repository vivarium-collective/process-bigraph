import numpy as np
import os
from typing import *
from abc import ABC, abstractmethod
import uuid
import datetime
#import matplotlib.pyplot as plt
#from scipy.io.wavfile import write
from process_bigraph.composite import Process, Composite
from process_bigraph.registry import process_registry


class SignalModulationProcess(Process):
    """Generic base class for signal modulators."""

    config_schema = {
        'input_signal': {
            '_type': 'list[float]',
            '_default': []
        },
        'duration': 'int'
    }

    def __init__(self, config=None):
        """These processes are more of steps, I suppose."""
        super().__init__(config)

    def initial_state(self):
        return {}

        '''return {
            'input_signal': self.config['input_signal'],
            'output_signal': []
        }'''

    def schema(self):
        return {
            'input_signal': 'list[float]',
            'output_signal': 'list[float]'
        }

    def update(self, state, interval):
        return {}


class MediumDistortionProcess(SignalModulationProcess):
    config_schema = {
        'input_signal': 'list[float]',
    }

    def __init__(self, config=None):
        super().__init__(config)

    def update(self, state, interval):
        input_signal = np.array(state['input_signal'])
        new_wave = apply_modulation(np.array(state['output_signal']), distortion, gain=5)
        wav_fp = 'distortion_' + str(datetime.datetime.utcnow()).replace(':', '').replace(' ', '').replace('.', '') + '.wav'
        '''array_to_wav(
            filename=os.path.join(
                os.getcwd(),
                wav_fp
            ),
            input_signal=new_wave
        )
        plot_signal(duration=self.config['duration'], signal=new_wave, plot_label=wav_fp, fp=wav_fp.replace('.wav', '.png'))'''
        return {
            'input_signal': input_signal.tolist(),
            'output_signal': new_wave.tolist()
        }


class TremoloProcess(SignalModulationProcess):
    config_schema = {
        'rate': 'int',
        'depth': 'float',
    }

    def __init__(self, config=None):
        super().__init__(config)

    def update(self, state, interval):
        input_signal = np.array(state['output_signal'])
        # create new wave
        new_wave_modulated = apply_modulation(
            input_wave=input_signal,
            modulation_function=tremolo,
            depth=self.config['depth'],
            rate=self.config['rate'],
        )
        print(new_wave_modulated)

        # write out the file
        wav_fp = 'tremolo_' + str(datetime.datetime.utcnow()).replace(':', '').replace(' ', '').replace('.', '') + '.wav'
        '''array_to_wav(
            filename=os.path.join(
                os.getcwd(),
                wav_fp
            ),
            input_signal=new_wave_modulated
        )
        plot_signal(self.config['duration'], signal=new_wave_modulated, plot_label=wav_fp)'''

        return {
            'input_signal': input_signal.tolist(),
            'output_signal': new_wave_modulated.tolist()
        }


class RingModulationProcess(SignalModulationProcess):
    config_schema = {
        'input_signal': 'list[float]',
        'mod_freq': 'int',
    }

    def __init__(self, config=None):
        super().__init__(config)

    def update(self, state, interval):
        input_signal = np.array(state['output_signal'])
        # create new wave
        new_wave_modulated = apply_modulation(
            input_wave=np.array(state['output_signal']),
            modulation_function=ring_modulation,
            mod_freq=self.config['mod_freq'],
        )

        # write out the file
        results_dir = os.path.join(os.getcwd(), 'ring_mod_results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        wav_fp = 'ring_mod_' + str(datetime.datetime.utcnow()).replace(':', '').replace(' ', '').replace('.', '') + '.wav'
        '''array_to_wav(filename=os.path.join(results_dir, wav_fp), input_signal=new_wave_modulated)
        plot_signal(duration=self.config['duration'], signal=new_wave_modulated, plot_label=wav_fp, fp=os.path.join(results_dir, wav_fp.replace('.wav', '.png')))'''
        return {
            'input_signal': input_signal.tolist(),
            'output_signal': new_wave_modulated.tolist()
        }


class PhaserProcess(SignalModulationProcess):
    config_schema = {
        'input_signal': 'list[float]',
        'rate': 'int',
        'depth': 'int'
    }

    def __init__(self, config=None):
        super().__init__(config)

    def update(self, state, interval):
        # create new wave
        input_signal = np.array(state['input_signal'])
        new_wave_modulated = apply_modulation(
            input_wave=np.array(state['output_signal']),
            modulation_function=phaser,
            rate=self.config['rate'],
            depth=self.config['depth']
        )

        # write out the file
        results_dir = os.path.join(os.getcwd(), 'phaser_results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        wav_fp = 'phaser_' + str(datetime.datetime.utcnow()).replace(':', '').replace(' ', '').replace('.', '') + '.wav'
        #array_to_wav(filename=os.path.join(results_dir, wav_fp), input_signal=new_wave_modulated)
        #plot_signal(duration=self.config['duration'], signal=new_wave_modulated, plot_label=wav_fp, fp=os.path.join(results_dir, wav_fp.replace('.wav', '.png')))
        return {
            'input_signal': input_signal.tolist(),
            'output_signal': new_wave_modulated.tolist()
        }


class DelayProcess(SignalModulationProcess):
    config_schema = {
        'input_signal': 'list[float]',
        'delay_time': 'float',
        'decay': 'float'
    }

    def __init__(self, config=None):
        super().__init__(config)

    def update(self, state, interval):
        # create new wave
        input_signal = np.array(state['input_signal'])
        new_wave_modulated = apply_modulation(
            input_wave=np.array(state['output_signal']),
            modulation_function=delay,
            delay_time=self.config['delay_time'],
            decay=self.config['decay']
        )

        # write out the file
        results_dir = os.path.join(os.getcwd(), 'delay_results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        wav_fp = 'delay_' + str(datetime.datetime.utcnow()).replace(':', '').replace(' ', '').replace('.', '') + '.wav'
        #array_to_wav(filename=os.path.join(results_dir, wav_fp), input_signal=new_wave_modulated)
        #plot_signal(duration=self.config['duration'], signal=new_wave_modulated, plot_label=wav_fp, fp=os.path.join(results_dir, wav_fp.replace('.wav', '.png')))
        return {
            'input_signal': input_signal.tolist(),
            'output_signal': new_wave_modulated.tolist()
        }


class PedalBoardProcess(SignalModulationProcess):
    config_schema = {
        'input_signal': 'list[float]',
        'pedals': 'tree[any]'
    }

    def __init__(self, config=None):
        super().__init__(config)

    def update(self, state, interval):
        # create new wave
        output_signal = []

        # TODO: sort this according to the desired pedal order in the chain
        for pedal_type, pedal_config in self.config['pedals'].items():
            modulated_signal = modulate_signal(
                instance_type=pedal_type,
                duration=self.config['duration'],
                instance_config=pedal_config
            )
            output_signal += modulated_signal

        # write out the file
        results_dir = os.path.join(os.getcwd(), 'ring_mod_results')
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        wav_fp = 'pedalboard_' + str(datetime.datetime.utcnow()).replace(':', '').replace(' ', '').replace('.', '') + '.wav'
        #array_to_wav(filename=os.path.join(results_dir, wav_fp), input_signal=output_signal)
        #plot_signal(duration=self.config['duration'], signal=output_signal, plot_label=wav_fp, fp=os.path.join(results_dir, wav_fp.replace('.wav', '.png')))
        return {
            'output_signal': output_signal
        }


process_registry.register('medium_distortion', MediumDistortionProcess)
process_registry.register('tremolo', TremoloProcess)
process_registry.register('ring_modulation', RingModulationProcess)
process_registry.register('phaser', PhaserProcess)
process_registry.register('delay', DelayProcess)
process_registry.register('pedalboard', PedalBoardProcess)


def apply_modulation(input_wave: np.ndarray, modulation_function, **kwargs) -> np.ndarray:
    """
    Apply a modulation effect to an input wave.

    :param input_wave: NumPy array, the input waveform.
    :param modulation_function: function, the modulation effect to apply.
    :param kwargs: additional keyword arguments for the modulation function.
    :return: NumPy array, the modulated waveform.
    """
    return modulation_function(input_wave, **kwargs)


def distortion(input_wave: np.ndarray, gain=1):
    """
    Apply a simple distortion effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param gain: float, the gain factor for distortion.
    :return: NumPy array, the distorted waveform.
    """
    return np.clip(input_wave * gain, -1, 1)


def tremolo(input_wave, rate=5, depth=0.5):
    """
    Apply a tremolo effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param rate: float, the rate of the tremolo effect.
    :param depth: float, the depth of the tremolo effect.
    :return: NumPy array, the waveform with tremolo effect.
    """
    t = np.linspace(0, 1, len(input_wave), endpoint=True)
    modulating_wave = (1 - depth) + depth * np.sin(2 * np.pi * rate * t)
    return input_wave * modulating_wave


def ring_modulation(input_wave: np.ndarray, mod_freq=30):
    """
    Apply a ring modulation effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param t: NumPy array, the time series
    :param mod_freq: float, the frequency of the modulating wave.
    :return: NumPy array, the waveform with ring modulation effect.
    """
    sample_rate = 44100
    t = np.linspace(0, 1, len(input_wave), endpoint=True)
    modulating_wave = np.sin(2 * np.pi * mod_freq * t)
    return input_wave * modulating_wave


def bit_crusher(input_wave: np.ndarray, bit_depth=8):
    """
    Apply a bit crusher effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param bit_depth: int, the target bit depth.
    :return: NumPy array, the waveform with bit crusher effect.
    """
    max_val = np.max(np.abs(input_wave))
    input_wave_normalized = input_wave / max_val
    step = 2 ** bit_depth
    crushed_wave = np.round(input_wave_normalized * step) / step
    return crushed_wave * max_val


def delay(input_wave, delay_time=0.2, decay=0.5, max_delay=1.0, fs=500):
    """
    Apply a simple delay (echo) effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param delay_time: float, the delay time in seconds.
    :param decay: float, the decay factor for the echoes.
    :param max_delay: float, the maximum delay time in seconds.
    :param fs: int, the sampling rate (samples per second).
    :return: NumPy array, the waveform with delay effect.
    """
    delay_samples = int(delay_time * fs)
    max_delay_samples = int(max_delay * fs)
    output_wave = np.zeros(len(input_wave) + max_delay_samples)
    for i in range(len(input_wave)):
        output_wave[i] += input_wave[i]
        if i + delay_samples < len(output_wave):
            output_wave[i + delay_samples] += input_wave[i] * decay
    return output_wave[:len(input_wave)]


def chorus(input_wave, depth=0.5, rate=2, mix=0.5, fs=500):
    """
    Apply a chorus effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param depth: float, the depth of the chorus modulation.
    :param rate: float, the rate of the chorus modulation.
    :param mix: float, the mix of the original and modulated signal.
    :param fs: int, the sampling rate (samples per second).
    :return: NumPy array, the waveform with chorus effect.
    """
    modulating_wave = depth * np.sin(2 * np.pi * rate * np.linspace(0, 1, len(input_wave)))
    output_wave = np.zeros_like(input_wave)
    for i in range(len(input_wave)):
        delay_samples = int(modulating_wave[i] * fs)
        if i + delay_samples < len(input_wave):
            output_wave[i] = (1 - mix) * input_wave[i] + mix * input_wave[i + delay_samples]
        else:
            output_wave[i] = input_wave[i]
    return output_wave


def phaser(input_wave, rate=1, depth=0.5, freq=0.5, fs=500):
    """
    Apply a phaser effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param rate: float, the rate of the phaser effect.
    :param depth: float, the depth of the phaser effect.
    :param freq: float, the frequency of the phaser effect.
    :param fs: int, the sampling rate (samples per second).
    :return: NumPy array, the waveform with phaser effect.
    """
    output_wave = np.copy(input_wave)
    phase = 0
    for i in range(len(input_wave)):
        phase += depth * np.sin(2 * np.pi * rate * i / fs)
        filter_freq = freq + freq * phase
        output_wave[i] = input_wave[i] * np.sin(2 * np.pi * filter_freq * i / fs)
    return output_wave


'''def array_to_wav(filename, input_signal, sample_rate=44100):
    """
    Writes a NumPy array to a WAV file.

    Parameters:
    input_signal (numpy.ndarray): The input signal (audio data).
    sample_rate (int): The sample rate of the audio (in Hz).
    filename (str): The name of the output WAV file.
    """
    # Normalize the signal to 16-bit integer range
    input_signal = np.array(input_signal)
    max_val = np.iinfo(np.int16).max
    normalized_signal = np.int16(input_signal / np.max(np.abs(input_signal)) * max_val)

    # Write to WAV file
    write(filename, sample_rate, normalized_signal)'''


def initialize_timepoints(duration: int, sample_rate=44100) -> np.ndarray:
    return np.linspace(start=0, stop=duration, num=int(sample_rate * duration), endpoint=True)


def start_sine_wave(duration: int, pitch_frequency: int = 440) -> np.ndarray:
    sample_rate = 44100
    t = np.linspace(start=0, stop=1, num=int(sample_rate * duration), endpoint=True)
    return 0.5 * np.sin(2 * np.pi * pitch_frequency * t)  # Example sine wave at 440 Hz


def adjust_pitch_frequency(starting_frequency: float, n_semitones: float) -> float:
    return starting_frequency * 2 ** (n_semitones / 12)


'''def plot_signal(duration: int, signal: np.ndarray, plot_label: str, fp: str, show=False):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)
    plt.figure(figsize=(12, 6))
    # plt.subplot(3, 1, 1)
    sample_rate = 44100
    t = np.linspace(start=0, stop=duration, num=len(signal), endpoint=True)
    plt.plot(t, signal, label=plot_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if show:
        plt.show()
    plt.savefig(fp)


def plot_multi_modulation(t, input_wave, distorted_wave, tremolo_wave):
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, input_wave, label='Original Wave')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, distorted_wave, label='Distorted Wave')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, tremolo_wave, label='Tremolo Wave')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()'''


# TODO: matmul for types?


def run_instance(instance, num_beats=4):
    # make the composite
    workflow = Composite({
        'state': instance
    })
    # run
    workflow.run(num_beats)
    # gather results
    return workflow.gather_results()


# def test_tremolo():
#     stop = 32
#     frequencies = [262, 294, 330, 349]
#     starting_signal = start_sine_wave(stop)

#     def tremolo_create_instance(starting_signal):
#         return {
#             'tremolo': {
#                 '_type': 'process',
#                 'address': 'local:tremolo',
#                 'config': {
#                     'depth': 0.9,
#                     'rate': 9,
#                     'duration': stop,
#                     'input_signal': starting_signal
#                 },
#                 'wires': {  # this should return that which is in the schema
#                     'output_signal': ['output_signal_store'],
#                 }
#             },
#             'emitter': {
#                 '_type': 'step',
#                 'address': 'local:ram-emitter',
#                 'config': {
#                     'ports': {
#                         'inputs': {
#                             'output_signal': 'list[float]'
#                         },
#                     }
#                 },
#                 'wires': {
#                     'inputs': {
#                         'output_signal': ['output_signal_store'],
#                     }
#                 }
#             }
#         }

#     def run_instance(instance, num_beats=stop):
#         # make the composite
#         workflow = Composite({
#             'state': instance
#         })

#         num_beats = 3
#         # run
#         workflow.run(num_beats)

#         # gather results
#         return workflow.gather_results()

#     instance = tremolo_create_instance(starting_signal)
#     result = run_instance(instance)


# def test_ring_mod():
#     duration = 8
#     pitch_frequency = 800
#     initial_signal = start_sine_wave(duration, pitch_frequency)

#     def ring_mod_create_instance():
#         return {
#             'ring_modulation': {
#                 '_type': 'process',
#                 'address': 'local:ring_modulation',
#                 'config': {
#                     'mod_freq': 2000,
#                     'input_signal': initial_signal,
#                     'duration': duration
#                 },
#                 'wires': {  # this should return that which is in the schema
#                     'output_signal': ['output_signal_store'],
#                 }
#             },
#             'emitter': {
#                 '_type': 'step',
#                 'address': 'local:ram-emitter',
#                 'config': {
#                     'ports': {
#                         'inputs': {
#                             'output_signal': 'list[float]'
#                         },
#                     }
#                 },
#                 'wires': {
#                     'inputs': {
#                         'output_signal': ['output_signal_store'],
#                     }
#                 }
#             }
#         }

#     instance = ring_mod_create_instance()
#     result = run_instance(instance, num_beats=8)[('emitter',)]
#     #array_to_wav(os.path.join('ring_mod_results', 'input_signal.wav'), input_signal=initial_signal)
#     #resulting_wave = np.array(result[('emitter',)])
#     print(len(result), type(result))
#     for r in result:
#         print(type(r))
#     #plot_signal(duration, resulting_wave, 'final_ring_mod_wave', fp='final_ring_mod_result')


# def test_phaser():
#     duration = 8
#     pitch_frequency = 800
#     rate = 3
#     depth = 0.75
#     initial_signal = start_sine_wave(duration, pitch_frequency)

#     def phaser_create_instance():
#         return {
#             'phaser': {
#                 '_type': 'process',
#                 'address': 'local:phaser',
#                 'config': {
#                     'input_signal': initial_signal,
#                     'rate': rate,
#                     'depth': depth,
#                     'duration': duration
#                 },
#                 'wires': {  # this should return that which is in the schema
#                     'input_signal': ['input_signal_store'],
#                     'output_signal': ['output_signal_store'],
#                 }
#             },
#             'emitter': {
#                 '_type': 'step',
#                 'address': 'local:ram-emitter',
#                 'config': {
#                     'ports': {
#                         'inputs': {
#                             'output_signal': 'list[float]'
#                         },
#                     }
#                 },
#                 'wires': {
#                     'inputs': {
#                         'output_signal': ['output_signal_store'],
#                     }
#                 }
#             }
#         }

#     instance = phaser_create_instance()
#     result = run_instance(instance, num_beats=8)[('emitter',)]
#     #array_to_wav(os.path.join('phaser_results', 'input_signal.wav'), input_signal=initial_signal)
#     #resulting_wave = np.array(result[('emitter',)])
#     print(len(result), type(result))
#     for r in result:
#         print(type(r))
#     #plot_signal(duration, resulting_wave, 'final_ring_mod_wave', fp='final_ring_mod_result')


# def test_delay():
#     duration = 8
#     b_flat = adjust_pitch_frequency(440.0, 1.0)
#     delay_time = 0.6
#     decay = 0.4
#     initial_signal = start_sine_wave(duration, b_flat)

#     def delay_create_instance():
#         return {
#             'phaser': {
#                 '_type': 'process',
#                 'address': 'local:delay',
#                 'config': {
#                     'input_signal': initial_signal,
#                     'delay_time': delay_time,
#                     'decay': decay,
#                     'duration': duration
#                 },
#                 'wires': {  # this should return that which is in the schema
#                     'output_signal': ['output_signal_store'],
#                 }
#             },
#             'emitter': {
#                 '_type': 'step',
#                 'address': 'local:ram-emitter',
#                 'config': {
#                     'ports': {
#                         'inputs': {
#                             'output_signal': 'list[float]'
#                         },
#                     }
#                 },
#                 'wires': {
#                     'inputs': {
#                         'output_signal': ['output_signal_store'],
#                     }
#                 }
#             }
#         }


# def create_instance(instance_type: str, wires: Dict[str, Any], **instance_config):
#     return {
#         instance_type: {
#             '_type': 'process',
#             'address': f'local:{instance_type}',
#             'config': instance_config,
#             'wires': wires
#         },
#         'emitter': {
#             '_type': 'step',
#             'address': 'local:ram-emitter',
#             'config': {
#                 'ports': {
#                     'inputs': {
#                         'output_signal': 'list[float]'
#                     },
#                 }
#             },
#             'wires': {
#                 'inputs': {
#                     'output_signal': ['output_signal_store'],
#                 }
#             }
#         }
#     }


# def create_modulation_instance(instance_type: str, **instance_config):
#     """Types to be expected are modeled in a meta-instance below:
#     ```
#          instance = {
#             'tremolo': {
#                 'config': {
#                     'input_signal': initial_signal,
#                     'rate': tremolo_rate,
#                     'depth': tremolo_depth,
#                     'duration': duration
#                 },
#             },
#             'ring_modulation': {
#                 'config': {
#                     'input_signal': 'output_signal_store',
#                     'mod_freq': ring_mod_freq,
#                     'duration': duration
#                 }
#             },
#             'delay': {
#                 'config': {
#                     'input_signal': 'output_signal_store',
#                     'delay_time': delay_time,
#                     'decay': decay,
#                     'duration': duration
#                 },
#             },
#     ```
#     """
#     wires = {'output_signal': ['output_signal_store']}
#     return create_instance(instance_type, wires, **instance_config)


# def modulate_signal(instance_type: str, duration: int, **instance_config):
#     instance = create_modulation_instance(instance_type, **instance_config)
#     result = run_instance(instance, duration)[('emitter',)]
#     return result[-1]['output_signal']


# def test_pedalboard_process():
#     """# set global parameters
#     duration = 8
#     b_flat = adjust_pitch_frequency(440.0, 1.0)
#     initial_signal = start_sine_wave(duration, b_flat)

#     # pedal-specific parameters
#     delay_time = 0.6
#     decay = 0.4
#     tremolo_rate = 3
#     tremolo_depth = 0.75
#     ring_mod_freq = 1000

#     # define pedal spec (must correspond to the process instance types)
#     pedals = {
#         'delay': {
#             'delay_time': delay_time,
#             'decay': decay
#         },
#         'tremolo': {
#             'rate': tremolo_rate,
#             'depth': tremolo_depth
#         },
#         'ring_modulation': {
#             'mod_freq': ring_mod_freq
#         }
#     }


#     instance = {
#         'pedalboard': {
#             '_type': 'process',
#             'address': 'local:pedalboard',
#             'config': {
#                 'input_signal': initial_signal,
#                 'duration': duration,
#                 'pedals': pedals,
#             },
#             'wires': {  # this should return that which is in the schema
#                 'input_signal': ['input_signal_store'],
#                 'output_signal': ['output_signal_store'],
#             }
#         },
#         'emitter': {
#             '_type': 'step',
#             'address': 'local:ram-emitter',
#             'config': {
#                 'ports': {
#                     'inputs': {
#                         'input_signal': 'list[float]',
#                         'output_signal': 'list[float]'
#                     },
#                 }
#             },
#             'wires': {
#                 'inputs': {
#                     'input_signal': ['input_signal_store'],
#                     'output_signal': ['output_signal_store'],
#                 }
#             }
#         }
#     }

#     result = run_instance(instance, num_beats=8)[('emitter',)]
#     #array_to_wav('input_signal.wav', input_signal=initial_signal)
#     #resulting_wave = np.array(result[('emitter',)])
#     final_result = result[-1]['output_signal']
#     #plot_signal(duration, final_result, 'final_wave', fp='final_composite_wave')"""
#     pass


# def test_pedalboard():
#     # set global parameters
#     duration = 8
#     b_flat = adjust_pitch_frequency(440.0, 1.0)
#     initial_signal = start_sine_wave(duration, b_flat)

#     # pedal-specific parameters
#     delay_time = 0.6
#     decay = 0.4
#     tremolo_rate = 3
#     tremolo_depth = 0.75
#     ring_mod_freq = 1000

#     instance = {
#         'ring_modulation': {
#             '_type': 'process',
#             'address': 'local:ring_modulation',
#             'config': {
#                 'mod_freq': 2000,
#                 'input_signal': initial_signal,
#                 'duration': duration
#             },
#             'wires': {  # this should return that which is in the schema
#                 'input_signal': ['input_signal_store'],
#                 'output_signal': ['output_signal_store'],
#             }
#         },
#         'tremolo': {
#             '_type': 'process',
#             'address': 'local:tremolo',
#             'config': {
#                 'depth': 0.9,
#                 'rate': 9,
#                 'duration': duration,
#             },
#             'wires': {  # this should return that which is in the schema
#                 'input_signal': 'input_signal_store',
#                 'output_signal': ['output_signal_store'],
#             }
#         },
#         'emitter': {
#             '_type': 'step',
#             'address': 'local:ram-emitter',
#             'config': {
#                 'ports': {
#                     'inputs': {
#                         'input_signal': 'list[float]',
#                         'output_signal': 'list[float]'
#                     },
#                 }
#             },
#             'wires': {
#                 'inputs': {
#                     'input_signal': ['input_signal_store'],
#                     'output_signal': ['output_signal_store'],
#                 }
#             }
#         }
#     }

#     result = run_instance(instance, duration)
#     print(f'RESULT: {result}')


if __name__ == '__main__':
    pass

    # test_pedalboard()
    # test_delay()
    # test_phaser()
    # test_ring_mod()
    # test_tremolo()
