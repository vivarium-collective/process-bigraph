import numpy as np
import os
import uuid
import datetime
from scipy.io.wavfile import write
from process_bigraph.composite import Process, Composite
from process_bigraph.registry import process_registry
# import matplotlib.pyplot as plt


class MediumDistortionProcess(Process):
    config_schema = {
        'input_signal': 'list[float]',
    }

    def __init__(self, config=None):
        super().__init__(config)

    def initial_state(self):
        return {'output_signal': self.config['input_signal']}

    def schema(self):
        return {
            'output_signal': 'list[float]',
        }

    def update(self, state, interval):
        new_wave = apply_modulation(np.array(state['output_signal']), distortion, gain=5)
        array_to_wav(
            filename=os.path.join(
                os.getcwd(),
                'distortion_' + str(datetime.datetime.utcnow()).replace(':', '').replace(' ', '').replace('.', '') + '.wav'
            ),
            input_signal=new_wave
        )
        return {
            'output_signal': new_wave.tolist()
        }


class TremoloProcess(Process):
    config_schema = {
        'input_signal': 'list[float]',
        'rate': 'int',
        'depth': 'float',
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.input_signal = self.config['input_signal']

    def initial_state(self):
        return {'output_signal': self.input_signal}

    def schema(self):
        return {
            'output_signal': 'list[float]',
        }

    def update(self, state, interval):
        # create new wave
        new_wave_modulated = apply_modulation(
            input_wave=np.array(state['output_signal']),
            modulation_function=tremolo,
            depth=self.config['depth'],
            rate=self.config['rate']
        )
        print(new_wave_modulated)

        # write out the file
        array_to_wav(
            filename=os.path.join(
                os.getcwd(),
                'tremolo_' + str(datetime.datetime.utcnow()).replace(':', '').replace(' ', '').replace('.', '') + '.wav'
            ),
            input_signal=new_wave_modulated
        )

        return {
            'output_signal': new_wave_modulated.tolist()
        }


class RingModulationProcess(Process):
    """These processes are more of steps, I suppose."""
    config_schema = {
        'input_signal': 'list[float]',
        'mod_freq': 'int',
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.input_signal = self.config['input_signal']

    def initial_state(self):
        return {'output_signal': self.input_signal}

    def schema(self):
        return {
            'output_signal': 'list[float]',
        }

    def update(self, state, interval):
        # create new wave
        new_wave_modulated = apply_modulation(
            input_wave=np.array(state['output_signal']),
            modulation_function=ring_modulation,
            mod_freq=self.config['mod_freq']
        )

        # write out the file
        array_to_wav(
            filename=os.path.join(
                os.getcwd(),
                'ring_mod_' + str(datetime.datetime.utcnow()).replace(':', '').replace(' ', '').replace('.', '') + '.wav'
            ),
            input_signal=new_wave_modulated
        )

        return {
            'output_signal': new_wave_modulated.tolist()
        }


process_registry.register('medium_distortion', MediumDistortionProcess)
process_registry.register('tremolo', TremoloProcess)
process_registry.register('ring_modulation', RingModulationProcess)


def apply_modulation(input_wave, modulation_function, **kwargs):
    """
    Apply a modulation effect to an input wave.

    :param input_wave: NumPy array, the input waveform.
    :param modulation_function: function, the modulation effect to apply.
    :param kwargs: additional keyword arguments for the modulation function.
    :return: NumPy array, the modulated waveform.
    """
    return modulation_function(input_wave, **kwargs)


def distortion(input_wave, gain=1):
    """
    Apply a simple distortion effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param gain: float, the gain factor for distortion.
    :return: NumPy array, the distorted waveform.
    """
    return np.clip(input_wave * gain, -1, 1)


def tremolo(input_wave, rate=5, depth=0.75):
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


def ring_modulation(input_wave, mod_freq=30):
    """
    Apply a ring modulation effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param mod_freq: float, the frequency of the modulating wave.
    :return: NumPy array, the waveform with ring modulation effect.
    """
    t = np.linspace(0, 1, len(input_wave/2), endpoint=True)
    modulating_wave = np.sin(2 * np.pi * mod_freq * t)
    return input_wave * modulating_wave


def bit_crusher(input_wave, bit_depth=8):
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


def array_to_wav(filename, input_signal, sample_rate=44100):
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
    write(filename, sample_rate, normalized_signal)


def start_sine_wave(duration: int, pitch_frequency: int = 440):
    sample_rate = 44100  # Sample rate in Hz
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * pitch_frequency * t)  # Example sine wave at 440 Hz


def adjust_pitch(starting_frequency, n_semitones):
    return starting_frequency * 2 ** (n_semitones / 12)


def run_instance(instance, num_beats=4):
    # make the composite
    workflow = Composite({
        'state': instance
    })

    # run
    workflow.run(num_beats)

    # gather results
    return workflow.gather_results()


'''# Example usage
t = np.linspace(0, 1, 500, endpoint=True)
input_wave = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500, endpoint=True))  # Example sine wave

# Apply distortion
modulated_wave = apply_modulation(input_wave, distortion, gain=5)
print(modulated_wave)
# Apply tremolo
modulated_wave = apply_modulation(modulated_wave, tremolo, rate=5, depth=0.7)
print(modulated_wave)'''


'''# Plotting
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


def test_medium_distortion():
    initial_signal = start_sine_wave(10).tolist()
    instance = {
            'distortion': {
                '_type': 'process',
                'address': 'local:medium_distortion',
                'config': {
                    'input_signal': initial_signal,
                },
                'wires': {  # this should return that which is in the schema
                    'output_signal': ['output_signal_store'],
                }
            },
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'ports': {
                        'inputs': {
                            'output_signal': 'list[float]'
                        },
                    }
                },
                'wires': {
                    'inputs': {
                        'output_signal': ['output_signal_store'],
                    }
                }
            }
        }

    # make the composite
    workflow = Composite({
        'state': instance
    })

    num_beats = 3
    # run
    workflow.run(num_beats)

    # gather results
    results = workflow.gather_results()
    #print(results)
    #array_to_wav(os.path.join(os.getcwd(), 'result.wav'))


def test_tremolo():
    stop = 4
    frequencies = [262, 294, 330, 349]

    def tremolo_create_instance(starting_signal):
        return {
            'tremolo': {
                '_type': 'process',
                'address': 'local:tremolo',
                'config': {
                    'depth': 0.9,
                    'rate': 9,
                    'starting_frequency': 300
                },
                'wires': {  # this should return that which is in the schema
                    'output_signal': ['output_signal_store'],
                }
            },
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'ports': {
                        'inputs': {
                            'output_signal': 'list[float]'
                        },
                    }
                },
                'wires': {
                    'inputs': {
                        'output_signal': ['output_signal_store'],
                    }
                }
            }
        }

    def run_instance(instance, num_beats=stop):
        # make the composite
        workflow = Composite({
            'state': instance
        })

        num_beats = 3
        # run
        workflow.run(num_beats)

        # gather results
        return workflow.gather_results()

    measure = []
    for f in frequencies:
        starting_signal = start_sine_wave(stop, f)
        instance = tremolo_create_instance(starting_signal)
        result = run_instance(instance)
        measure.append(result)

    '''instance = {
            'distortion': {
                '_type': 'process',
                'address': 'local:tremolo',
                'config': {
                    'input_signal': initial_signal,
                },
                'wires': {  # this should return that which is in the schema
                    'output_signal': ['output_signal_store'],
                }
            },
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'ports': {
                        'inputs': {
                            'output_signal': 'list[float]'
                        },
                    }
                },
                'wires': {
                    'inputs': {
                        'output_signal': ['output_signal_store'],
                    }
                }
            }
        }'''

    '''# make the composite
    workflow = Composite({
        'state': instance
    })

    num_beats = 3
    # run
    workflow.run(num_beats)

    # gather results
    results = workflow.gather_results()
    print(results)
    #array_to_wav(os.path.join(os.getcwd(), 'result.wav'))'''


def test_ring_mod():
    stop = 10
    frequencies = [262, 294, 330, 349]

    def ring_mod_create_instance():
        return {
            'ring_modulation': {
                '_type': 'process',
                'address': 'local:ring_modulation',
                'config': {
                    'mod_freq': 2000,
                    'input_signal': start_sine_wave(stop)
                },
                'wires': {  # this should return that which is in the schema
                    'output_signal': ['output_signal_store'],
                }
            },
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'ports': {
                        'inputs': {
                            'output_signal': 'list[float]'
                        },
                    }
                },
                'wires': {
                    'inputs': {
                        'output_signal': ['output_signal_store'],
                    }
                }
            }
        }

    instance = ring_mod_create_instance()
    result = run_instance(instance, num_beats=8)
    resulting_wave = np.array(result[('emitter',)])
    print(type(resulting_wave))
   # array_to_wav(filename=os.path.join(os.getcwd(), 'final_result.wav'), input_signal=resulting_wave)


if __name__ == '__main__':
    test_ring_mod()
    # test_tremolo()
    # test_medium_distortion()
